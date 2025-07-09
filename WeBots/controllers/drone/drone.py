"""drone controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import math
import socket
import threading
import json
from collections import deque
from threading import Event
import base64
from PIL import Image
import cv2, numpy as np
import io




commands = deque()
# socket server
resolved_Event = Event()
resolved_Event.set()

cam_image = None


def start_command_server():
    global commands
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 9999))
    server.listen(1)
    
    while True:
        print("[Socket] waiting for command connection...")
        conn, addr = server.accept()
        print(f"[Socket] Connected to {addr}")
        
        while True:
            try:
                data = conn.recv(1024)
                if not data: 
                    break
                command = json.loads(data.decode())
                print(f"[Socket] Received: {command}")
                
                commands.append(command)
                resolved_Event.clear()
                resolved_Event.wait()
                
                type = command["type"]
                goal = command["goal"]
                
                response = None
                if type == "camera":
                    global cam_image
                    response = cam_image
                    response = (json.dumps(response, separators=(',', ':'), ensure_ascii=False) + "\n").encode("utf-8")
           
                else:
                    msg = f"Drone has successfully {type} by {goal}"
                    response = {"message": msg, "status": "done"}
                    response = (json.dumps(response) + "\n").encode("utf-8")
                data_out = response

                try:
                    conn.sendall(data_out)
                    print(f"fulfilled request of {type}")
                except BrokenPipeError:
                    print("[Socket] Client closed connection before we could send.")

            except ConnectionResetError:
                 print("[Socket] Connection lost unexpectedly")
                 break
                 
        conn.close()
    
# Start the thread
threading.Thread(target=start_command_server, daemon=True).start()
        

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

# Gives roll/pitch/yaw
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

# for angular velocity
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")

# motors
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(1.0)
    
k_vertical_thrust = 68.5;  # Base thrust to hover
k_vertical_offset = 0.6;   # Offset for altitude correction
k_vertical_p = 3.0;        # PID P value for vertical motion
k_roll_p = 50.0;           # Roll stabilisation
k_pitch_p = 30.0;          # Pitch stabilisation
target_altitude = 1.0      # Altitude drone should try to maintain


executing = False
target = None
initial_pos = None
distance = None
DIST_THRESHOLD = 0.7
ANGLE_THRESHOLD = math.radians(5)
COMMAND = ""


def meetsGoal():
    global COMMAND
    global initial_pos
    global distance
    if COMMAND == "forward":
        return meetsDistanceThreshold(gps.getValues(), initial_pos, distance, DIST_THRESHOLD)
    elif COMMAND == "right_rotate" or COMMAND == "left_rotate":
        roll, pitch, yaw = imu.getRollPitchYaw()
        return meetsAngleThreshold(yaw, target, ANGLE_THRESHOLD)
        

def meetsDistanceThreshold(pos1, pos2, distance, threshold):
    diff = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos1[2]) ** 2)
    if abs(diff  - distance) <= threshold:
        return True
   
    return False
        
        
def meetsAngleThreshold(yaw, target, threshold):
    yaw_error = abs(yaw - target)
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
    if abs(yaw_error) <= threshold:
        return True
        
    return False
    
def getImage(camera):
    raw = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()

    # 2) turn it into a NumPy array and drop alpha
    arr = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
    bgr = arr[:, :, :3]

    # 3) encode to JPEG (much smaller than raw RGBA!)
    success, jpg_buf = cv2.imencode('.jpg', bgr)
    if not success:
        raise RuntimeError("JPEG encoding failed")

    # 4) Base64-encode the JPEG bytes
    b64_str = base64.b64encode(jpg_buf.tobytes()).decode('ascii')

    # 5) build your JSON-serializable dict
    data = {
      "type": "camera",
      "image": {
        "format": "jpeg",
        "b64": b64_str,
        "height": h,
        "width": w
      }
    }
    
    global cam_image
    cam_image = data      
    
    
    


# Main loop:
while robot.step(timestep) != -1:
    
    time = robot.getTime()
    
    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]
    roll_velocity = gyro.getValues()[0]
    pitch_velocity = gyro.getValues()[1]
    
    camera_roll_motor.setPosition(-0.115 * roll_velocity)
    camera_pitch_motor.setPosition(-0.1 * pitch_velocity)
    
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    
    if executing == True:
        result = meetsGoal()
       
        if result:
            # can receive more commands from client
            print("results achieved! Stop everything!")
            resolved_Event.set()
            executing = False
            target = None
            COMMAND = ""
            distance = None
            initial_pos = None
        
        
    
    if len(commands) > 0 and executing == False:
        cur_command = commands.popleft()
        type = cur_command["type"]

        r, p, y = imu.getRollPitchYaw()
        initial_pos = gps.getValues()

        if type == "forward":
            # get distance
            dist = cur_command["goal"]
            
            # calculate target coords
            fx = math.cos(p) * math.cos(y)
            fy = math.cos(p) * math.sin(y)
            fz = math.sin(p)
            initial_pos = gps.getValues()
            distance = dist
            target = [fx * distance + initial_pos[0],
                  fy * distance + initial_pos[1],
                  fz * distance + initial_pos[2]]
            print(f"target location: {target}")
        elif type == "right_rotate" or type == "left_rotate":
            # get degree rotation
            degree = cur_command["goal"]
            
            # calculate target yaw
            target = yaw - math.radians(degree)
            
            target = (target + math.pi) % (2 * math.pi) - math.pi
            print(f"current yaw: {y}")
            print(f"taget yaw: {target}")
        
        
        
        if type == "camera":
            getImage(camera)
            resolved_Event.set()
        else:
            COMMAND = type
            executing = True
           
    

    # Executes command if it exists
    if COMMAND != "" and executing == True:
        if COMMAND == "forward":
            pitch_disturbance = -2.0        
        elif  COMMAND == "right_rotate":
            yaw_disturbance = -0.3
        elif COMMAND == "left_rotate":
            yaw_disturbance = 0.3
        elif COMMAND == "up":
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} m")
        elif COMMAND == "down":
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude:.2f} m")
 
          
        
    
    roll_input = k_roll_p * max(-1.0, min(1.0, roll)) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * max(-1.0, min(1.0, pitch)) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance
    altitude_error = target_altitude - altitude + k_vertical_offset
    clamped_altitude_error = max(-1.0, min(1.0, altitude_error))
    vertical_input = k_vertical_p * (clamped_altitude_error ** 3)

    # === Final Motor Commands ===
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)
        
   
    
        

    

# Enter here exit cleanup code.
