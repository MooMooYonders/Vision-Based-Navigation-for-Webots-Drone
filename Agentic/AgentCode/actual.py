import socket
import json
import matplotlib.pyplot as plt
import numpy as np
import base64
import cv2
import sys
from pathlib import Path
import torch
import pillow_heif
from PIL import Image
from ultralytics import YOLO
import pillow_heif
import time
from segment_anything import SamPredictor, sam_model_registry
import math
from transformers import SamModel, SamProcessor
import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.config import RunnableConfig
from openai import OpenAI
import io
from langgraph.types import Command
from pydantic import BaseModel, Field
from collections import defaultdict
from IPython.display import Image as ipyImage, display



depth_anything_path = Path("../Depth-Anything-V2/metric_depth")
sys.path.append(str(depth_anything_path))

from depth_anything_v2.dpt import DepthAnythingV2

# openai api key
load_dotenv()
api_key = os.getenv("API_KEY")


# establish the connection with webots
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 9999))

# socket commands for the drone control in webots
def recv_full_message(sock, delim=b'\n'):
    """
    Read from `sock` until we hit the delimiter at the end of the buffer.
    Returns the full message including the delimiter.
    """
    buf = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            # connection closed
            break
        buf.extend(chunk)
        if buf.endswith(delim):
            break
    return bytes(buf)

# gets image from camera
def camera_command(command="camera"):

    payload = {"type": command, "goal": "lol"}
    client.sendall(json.dumps(payload).encode())

    raw = recv_full_message(client)
    if not raw:
        raise ConnectionError("Server closed the connection")
    # Decode and parse JSON
    text = raw.decode("utf-8")
    text = text.rstrip("\n")
    response = json.loads(text)
    
    response = decodeImage(response)

    return response

# moves the drone by the specified command
def movement_command(command, goal):
    payload = {"type": command, "goal": goal}
    client.sendall(json.dumps(payload).encode())


    raw = recv_full_message(client)
    raw = raw.decode("utf-8")
    raw = raw.rstrip("\n")
    response = json.loads(raw)

    if response.get("status") != "done":
        raise RuntimeError(f"Unexpected reply {response}")

    return response

def decodeImage(payload):
    jpg_bytes = base64.b64decode(payload['image']['b64'])
    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    # BGR → RGB for matplotlib
    img = img[:, :, ::-1]
    return img


# ------ALL THE MODELS------

# DEPTH-ANYTHING MODEL
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitb' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
depth_model.load_state_dict(torch.load(f'{str(depth_anything_path)}/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
# important otherwise huge errors
depth_model.to(DEVICE)
depth_model.eval()

# YOLO-WORLD MODEL
yolo_model = YOLO("yolov8l-world.pt")

# SAM 2
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam2_model = SamModel.from_pretrained("facebook/sam-vit-large")
sam2_model.eval()


# ------MODEL RESULTS GENERATORS------


def depth_anything(model, array):
    start = time.time()
    results = model.infer_image(array)
    end = time.time()
    latency_ms = (end - start) * 1000 
    return results, latency_ms

def yolo_world(model, array, classes):
    start = time.time()
    model.set_classes(classes)
    results = model.predict(array)
    end = time.time()
    latency_ms = (end - start) * 1000 
    return results, latency_ms

def sam_2(image, box):
    start = time.time()
    image_pil = Image.fromarray(image)
    
    box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

    inputs = processor(
      images=image_pil,
      input_boxes=[[box]],           
      return_tensors="pt"
    )

    with torch.no_grad():
        outputs = sam2_model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0]

    end = time.time()
    latency_ms = (end - start) * 1000

    return masks[0].squeeze(0), latency_ms

# AGENT FUNCTIONS

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 240
horizontal_fov = 45
vertical_fov = 43.6

# calculate the distance from drone to object
def getDirections(pixel_coords,
                  estimated_depth,
                  horizontal_fov=horizontal_fov,
                  vertical_fov=vertical_fov,
                  image_width = IMAGE_WIDTH,
                  image_height = IMAGE_HEIGHT):
    half_width = int(image_width / 2)
    half_height = int(image_height / 2)
    vertical_angle = (pixel_coords[0] - half_height) / half_height * vertical_fov
    horizontal_angle = (pixel_coords[1] - half_width) / half_width * horizontal_fov
    print(f"vert ang: {vertical_angle}")
    print(f"hor ang: {horizontal_angle}")
    direction_vector = np.array([math.tan(math.radians(horizontal_angle)),
                        math.tan(math.radians(vertical_angle)),
                        1])
    actual_direction_vector = direction_vector / np.linalg.norm(direction_vector) * estimated_depth
    x_mov = actual_direction_vector[0]
    z_mov = actual_direction_vector[2]
    diagonal_perpendicular_distance = math.sqrt(z_mov ** 2 + x_mov ** 2)
    return actual_direction_vector[0], actual_direction_vector[1], actual_direction_vector[2], horizontal_angle, vertical_angle, diagonal_perpendicular_distance

# calculate movements from drone to object
def calculateMovementsToObjects(image, objects: list[str]):
    """using the image and object name provided, calculates the drone movements needed to reach the object"""
    depth, _ = depth_anything(depth_model, image)
    results, _ = yolo_world(yolo_model, image, objects)

    info = defaultdict(list)

    # predictor.set_image(array)
    for result in results:
        xyxy = result.boxes.xyxy
        names = result.names
        cls = result.boxes.cls
        
        for j in range(len(xyxy)):
            box_xyxy = xyxy[j].cpu().numpy()

            mask, _ = sam_2(image, box_xyxy.tolist())

            if not mask.any():
                continue
            
            depth_values = depth[mask]
            if not (depth_values > 0).any():
                continue

            depth_min = np.min(depth_values[depth_values > 0])
        
            # find the centroid of the object
            ys, xs = np.where(mask)
            centroid_y = ys.mean()
            centroid_x = xs.mean()

            # estimated dimensions
            width = box_xyxy[2] - box_xyxy[0]
            height = box_xyxy[3] - box_xyxy[1]

            # name of object
            name = names[int(cls[j].item())]

            x, y, z, yaw_rot, pitch_rot, diag_perpen_dist  = getDirections((centroid_y,centroid_x), estimated_depth=depth_min)
            print(f"Object identified: {name}")
            print(f"Relative coordinates of {name} is ({x}, {y}, {z})")
            print(f"Yaw rot to face {name} is {yaw_rot}")
            print(f"Pitch rot to face {name} is {pitch_rot}")
            print(f"Forward distance after yaw rot is {diag_perpen_dist}")
            print(f"Estimated width and height of {name} is {width}, {height}")
            
            info[name].append({
                "x": float(round(x, 2)),
                "y": float(round(y, 2)),
                "z": float(round(z, 2)),
                "yaw_rot": float(round(yaw_rot, 2)),
                "pitch_rot": float(round(pitch_rot, 2)),
                "forward_distance_after_yaw_rot": float(round(diag_perpen_dist, 2)),
                "width": float(round(width, 2)),
                "height": float(round(height, 2))
            })
                
    if not info:
        return None
    else:
        return info

# rotate drone
@tool
def rotateDrone(tool_call_id: Annotated[str, InjectedToolCallId], degree: float):
    "rotates the drone by the specified degree AND WAITS FOR COMPLETION"
    command = "left_rotate" if degree <= 0 else "right_rotate"
    response = movement_command(command, degree - 6)
    print(response["message"])

    msg = f"Successfully rotated the drone by {degree} degrees"

    return Command(
        update={
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
        }
    )

def rotateDrone_raw(degree):
    "rotates the drone by the specified degree AND WAITS FOR COMPLETION"
    command = "left_rotate" if degree <= 0 else "right_rotate"
    response = movement_command(command, degree)
    print(response["message"])
    return f"rotating the drone by {degree}"


# get the image of the drone
def getDroneImage():
    """gets the image in the camera of the drone"""
    img = camera_command()
    print("Got the image from the drone")
    return img

def getDroneImage_raw():
    """gets the image in the camera of the drone"""
    img = camera_command()
    print("Got the image from the drone")
    return img


# get the drone to move forward by x distance
@tool
def moveDroneForward(tool_call_id: Annotated[str, InjectedToolCallId], distance):
    "moves the drone forward by distance specified"
    response = movement_command("forward", distance - 0.6)
    print(response["message"])
    msg = f"Successfully moved the drone forward by {distance} m"

    return Command(
        update={
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
        }
    )


# check if the object is in the correct picture
openai_client = OpenAI(api_key=api_key)

def convertDroneImageTob64(image):
    pil_image = Image.fromarray(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return b64 
    

@tool
def checkObject(tool_call_id: Annotated[str, InjectedToolCallId], object: str):
    """checks if the object specified is captured by the drone's camera AND WAITS FOR COMPLETION"""
    image = getDroneImage()
    b64 = convertDroneImageTob64(image)
    data_uri = f"data:image/png;base64,{b64}"   


    prompt = f"Here is an image. Does it contain {object}? Reply with exactly `yes` or `no`"

    response = openai_client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"{data_uri}"},
                ],
            }
        ],
    )

    answer = response.output_text.lower().strip()
    msg = f"Yes, we have found the {object}" if answer == "yes" else f"No, we have not found the {object}"
    print(msg)

    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]
        }
    )

def checkObject_raw(object: str):
    """checks if the object specified is in the image"""
    image = getDroneImage()
    b64 = convertDroneImageTob64(image)
    data_uri = f"data:image/png;base64,{b64}"   


    prompt = f"Here is an image. Does it contain {object}? Reply with exactly `yes` or `no`"

    response = openai_client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"{data_uri}"},
                ],
            }
        ],
    )

    answer = response.output_text.lower().strip()
    return answer

class IdentifiedObjects(BaseModel):
    objects: list[str] = Field(
        description="All identified objects in the image that are relevant according to the prompt"
    )

def surveySurroundings(object: str):
    """Gets information about the object's position relative to the drone, as well as any other identified obstacles that are relevant"""

    image = getDroneImage()
    b64 = convertDroneImageTob64(image)
    data_uri = f"data:image/png;base64,{b64}"   

    prompt = f"""
    Here is an image from a drone camera and the main destination is {object}.

    You are to identify:
    - Any obstacles that may be blocking the path to reach the {object} 
    - Any other relevant objects that are close to the {object}

    Rules:
    - Only describe the objects themselves (e.g. bottle), something like `airplane next to bottle` is not accepted
    - You may use descriptive words for any object identified IF NECESSARY (e.g. GREEN bottle)
    - DO NOT input anything other than objects, something like `path seems clear` is not accepted
    """

    response = openai_client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"{data_uri}"},
                ],
            }
        ],
        text_format=IdentifiedObjects
    )

    if not response.output_parsed:
        return None

    identifiedObjects = response.output_parsed.model_dump()
    identifiedObjects = identifiedObjects["objects"]
    identifiedObjects.append(object)
    print(f"identified objects are {identifiedObjects}")

    result = calculateMovementsToObjects(image, identifiedObjects)

    return result


# AGENT CODE


# interprete
# ->
# Navigational Agent
  # plan the path with steps
    # takes in the image, finds object
    # gets info about object and any relevant obstacles
      # has access to detect object function -> gets the location, size of object
    # plans the path with steps to take
  # select the next step to take
# exectuor agent (one instance for one step)
  # at the end of each agent, connect it to obstacle checker agent to see if there is any new obstacles blocking
    # if there is, send back to planner to replan the steps




# ---STATES---
class InterpreterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]    # tracks all messages in agent system
    destination: str | None     


# ---OUTPUT AND MODELS---

# PLANNER

# planner -> returns a list of subtasks

class Step(BaseModel):
    explanation: str
    function: Literal["rotateDrone", "moveDroneForward"] = Field(
        ...,
        description="The function name that you wish to call"
    )
    argument: float

class MovementsReasoning(BaseModel):
    steps: list[Step]


class PlannerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]    
    destination: str | None
    object_details: dict

class PlannerOutput(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: list[dict] | None
    

def planner(state: PlannerState) -> PlannerOutput:
    message = state["messages"][0].content
    object_details = state["object_details"]
    destination = state["destination"]
    
    if not destination:
        return {
            "messages": [AIMessage(content="Missing destination information")]
        }
    if not object_details:
        return {
            "messages": [AIMessage(content="Missing object_details information")]
        }

    prompt = f"""
    You are a planning agent that outputs a JSON list of steps for the drone.  

    Rules:
    • Distances are in meters, as floats (e.g. 0.95) 2d.p.  
    • Rotations are in degrees (positive=right, negative=left).  
    • For each step, include a one-sentence math explanation justifying the function and its arguments.  

    Functions:
    - rotateDrone; takes in an angle as argument
        - if you want it to rotate right, put a positive argument, if you want it to rotate left, put a negative argument
    - moveDroneForward; takes in a distance as argument

    Destination:
    {destination}

    Information:
    - You are given an image from the drone's camera for reference
    - You are also given a json-formatted information in the form of a dictionary 
        - Each key represents the name of the object
        - Each value is a list of dictionaries where each dictionary represents one instance of the object that appears in the image. Each dictionary is represented as follows:
            - x represents the horizontal distance of the object relative to the drone (positive=right, negative=left)
            - y represents the vertical distance of the object relative to the drone (positive=down, negative=up)
            - z represents the forward distance of the object relative to the drone (positive=forward, negative=backward)
            - yaw_rot is the angle needed to rotate to face the object (positive=right rotate, negative=left rotate)
            - pitch_rot is the angle needed to rotate to face the object (positive=down rotate, negative=up rotate)
            - forward_distance_after_yaw_rot refers to the distance needed to move forward after the drone has yaw_rotated to face it
            - width refers to the estimated width of the object
            - height refers to the estimated height of the object

    Here is the information:
    {json.dumps(object_details, indent=2)}

    From the image reference, as well as the estimated details of some objects in the image, 
    you MUST plan a series of steps for the drone to reach the target destination

    These are additional instructions from the manager agent:
    {message}
    """

    image = getDroneImage()
    b64 = convertDroneImageTob64(image)
    data_uri = f"data:image/png;base64,{b64}"   

    response = openai_client.responses.parse(
        model="gpt-4o",
        input=[                      # this stays the same
        {
            "role": "user",
            "content": [
            {"type": "input_text",  "text": prompt},
            {"type": "input_image", "image_url": data_uri},
            ],
        }
        ],
        text_format=MovementsReasoning,
    )

    reasoning = response.output_parsed
    reasoning = reasoning.model_dump()["steps"] # list of step

    msg = "Planner: Successfully planned the steps for the drone"
    print(msg)

    return {
        "messages": [AIMessage(content=msg)],
        "steps": reasoning
    }

planner_graph = StateGraph(PlannerState, output=PlannerOutput)
planner_graph.add_node("planner", planner)

planner_graph.set_entry_point("planner")
planner_graph.add_edge("planner", END)

planner_subgraph = planner_graph.compile()


# EXECUTOR
class ExecutorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    step: dict

class ExecutorOutput(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

executor_tools = [rotateDrone, moveDroneForward]

executor_model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(executor_tools)

def should_executor_continue(state: ExecutorState):
    msg = state["messages"][-1]

    if msg.tool_calls:
        return "continue"
    
    return "end"

def executor(state: ExecutorState) -> ExecutorOutput:
    if not state["step"]:
        msg = "No step information available!"
        return {
            "messages": [AIMessage(content=msg)]
        }
    
    message = state["messages"][0].content
    function = state["step"]["function"]
    argument = state["step"]["argument"]

    prompt = f"""
    You are an executor agent executing a single step instruction

    You have access to the following tools:
    - rotateDrone, which takes a degree as argument
    - moveDroneForward, which takes a distance as argument

    You are to call the {function} function with the argument {argument}

    These are additional instructions from the manager agent:
    {message}
    """

    system_prompt = SystemMessage(content=prompt)

    response = executor_model.invoke([system_prompt])

    print(f"Executor: {response.content}")

    return {
        "messages": [AIMessage(content=response.content)]
    }


executor_graph = StateGraph(ExecutorState, output=ExecutorOutput)
executor_graph.add_node("executor", executor)
executor_graph.add_node("executor_tools", ToolNode(executor_tools))

executor_graph.set_entry_point("executor")

executor_graph.add_conditional_edges(
    "executor",
    should_executor_continue,
    {
        "continue": "executor_tools",
        END: END
    }
)

executor_subgraph = executor_graph.compile()

# PERCEPTOR

# perceptor agent, has access to getimage, rotate drone, yolo world and depth anything
# findobject -> rotates until found
# get details -> 

class PerceptorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]    
    destination: str | None
    object_details: dict | None
    next_agent: str | None

class PerceptorOutput(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]    
    object_details: dict | None


class PerceptorSelection(BaseModel):
    next_agent: Literal["locator", "scouter", "NA"] = Field(
        ...,
        description="The next agent to be used, or NA if no agent needs to be called"
    )
    explanation: str = Field(
        description="A one-sentence explanation for your choice of next agent or why no agent needs to be called"
    )

# tools
locator_tools = [rotateDrone, checkObject]

# models
perceptor_model = ChatOpenAI(model="gpt-4o", api_key=api_key).with_structured_output(PerceptorSelection)
locator_model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(locator_tools)

def perceptor(state: PerceptorState):
    prompt = """
    You are a perceptor agent that needs to do two things:
    - Locate the destination object
    - Get details about the destination's coordinates, as well as any object close to the destination

    You have access to two agents:
    - Locator: This agent would help you locate the destination object by rotating the drone until it finds it
    - Scouter: This agent would provide you with all the relevant positional details about the destination and any objects close to the destination object, ONLY CALL THIS AGENT ONCE YOUR DRONE FINDS THE OBJECT
        - Scouter will report back object_details if found in the form of json, that's when you know it has done its task

    Once you have fulfilled your tasks, return NA
    """
    system_prompt = SystemMessage(content=prompt)
    response = perceptor_model.invoke([system_prompt] + state["messages"])

    msg = f"""
    Perceptor:
    Next agent: {response.next_agent}
    Explanation: {response.explanation}
    """

    print(msg)

    return {
        "messages": [AIMessage(content=msg)],
        "next_agent": response.next_agent
    }


def scouter(state: PerceptorState):
    info = surveySurroundings(state["destination"])

    if not info:
        msg = "I can't seem to locate the destination object or any other surrounding objects, perhaps time to find the object again"
        print(f"Scouter: {msg}")
        return {
            "messages": [AIMessage(content=msg)],
            "object_details": None
        }
    
    msg = f"""
    Scouted information regarding the destination object and any other surrounding objects:
    {json.dumps(info, indent=2)}
    """
    print(f"Scouter: {msg}")

    return {
        "messages": [AIMessage(content=msg)],
        "object_details": info
    }
    

def locator(state: PerceptorState):
    obj = state["destination"]
    prompt = f"""
    You are a an object finder that needs to locate the destination object: {obj}

    You have access to the following tools:
    - rotateDrone tool with arguments: `degrees` which will rotate the drone by degrees specified
    - checkObject tool with argument: `object` to see if the object has been found

    You MUST use the available tools to locate the red car. Follow these steps:
    1. Check if the red car is visible (use checkObject tool)
    2. Rotate the drone (use rotateDrone tool)
    3. Repeat until you have found the destination object: {obj}

    NOTE that the drone's horizontal field of view is estimated to be around 60 degrees
    """

    system_prompt = SystemMessage(content=prompt)
    response = locator_model.invoke([system_prompt] + state["messages"])

    msg = "Locator fired!"
    print(msg)

    return {
        "messages": [response]
    }

# conditional edges
def check_located(state: PerceptorState):
    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "continue"
    
    return "perceptor"

def should_continue(state: PerceptorState):
    next_agent = state["next_agent"]

    if next_agent == "NA":
        return END
    
    return next_agent

# build graph
perceptor_graph = StateGraph(PerceptorState, output=PerceptorOutput)
perceptor_graph.add_node("perceptor", perceptor)
perceptor_graph.add_node("locator", locator)
perceptor_graph.add_node("locator_tools", ToolNode(locator_tools))
perceptor_graph.add_node("scouter", scouter)

perceptor_graph.add_edge(START, "perceptor")
perceptor_graph.add_edge("scouter", "perceptor")
perceptor_graph.add_edge("locator_tools", "locator")
perceptor_graph.add_conditional_edges(
    "locator",
    check_located,
    {
        "continue": "locator_tools",
        "perceptor": "perceptor"
    }
)
perceptor_graph.add_conditional_edges(
    "perceptor",
    should_continue,
    {
        "locator": "locator",
        "scouter": "scouter",
        END: END
    }
)

perceptor_subgraph = perceptor_graph.compile()


# INTERPRETER

# interpreter output
class DestinationResponse(BaseModel):
    """Information regarding user's intended destination"""

    can_find: Literal["yes", "no"] = Field(
        ...,
        description="Whether you were able to find the user's intended destination"
    )
    destination: str = Field(
        default="NA",
        description="The identified destination"
    )
    explanation: str = Field(
        description="A one-sentence justification for your conclusions about finding the destination or not"
    )

# interpreter model
interpreter_model = ChatOpenAI(model="gpt-4o", api_key=api_key).with_structured_output(DestinationResponse)

# FUNCTIONS

# interpreter function
def interpreter(state: InterpreterState):
    prompt = """
    You are an interpreter agent that extracts out the OBJECT/DESTINATION that the user wishes to go to.

    You would be given a statement from the user and you must extract out the user's intended destination object. 

    The expected output is as follows:
    - you must indicate whether or not you are able to identify the object destination with either `yes` or `no`
    - If the destination can be found, state just the destination, if not then reply with `NA`
    - Give a one-sentence explanation justifying why you think you can or can not identify the user's destination

    Here are some guiding questions:
    - Is there a place the user wishes to go to
    - Is that place associated with a object or a location
    - If the above two points are fulfilled, that is probably the identified object/destination

    The destination object can be descriptive e.g. green bottle, large orange statue
    """
    system_prompt = SystemMessage(content = prompt)
    response = interpreter_model.invoke([system_prompt] + state["messages"])
    
    msg = (f"""
    Interpreter:
    Able to find destination: {response.can_find}
    Identified destination: {response.destination}
    Explanation: {response.explanation}
    """)
    print(msg)

    return {
        "messages": [AIMessage(content=msg)],
        "destination": response.destination,
        "is_found": False
    }

    """
    if response.can_find:
        return Command(
            update={
                "messages": [AIMessage(content=msg)],
                "destination": response.destination},
            goto={"manager"}
            )
    else:
        msg += "\n Need to clarify user on destination details"
        return Command(
            update={"messages": [AIMessage(content=msg)]},
            goto="clarifier"
        )
    """


# MANAGERS

# overall state
class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    destination: str | None
    steps: list[dict] | None
    object_details: dict | None
    is_found: bool

# plan manager
class PlanManagerSelection(BaseModel):
    next_agent: Literal["perceptor_agent", "planner_agent", "step_manager"] = Field(
        ...,
        description="The next agent you wish to select"
    )
    explanation: str = Field(
        ...,
        description="A brief explanation for choosing the agent"
    )
    instructions: str = Field(
        None,
        description="Any extra instructions you wish to pass to the agent"
    )

manager_model = ChatOpenAI(model="gpt-4o", api_key=api_key).with_structured_output(PlanManagerSelection)


def PlanManager(state: OverallState) -> Command[Literal["perceptor_agent", "planner_agent", "step_manager"]]:
    prompt = """
    You are the planner manager agent 

    End Goal:
    You need to make sure a series of steps are drafted for the drone to reach the destination

    You have access to three agents:
    - Perceptor Agent: This agent will rotate the drone until it finds the target, and then report back the target destination's positional details, along with any other obstacles around its vicinity to you in the form of a dictionary
    - Planner Agent: This agent will take the object's positional details and return with a series of steps to take for the drone to reach the target destination

    If you believe you have achieved your objectives, then pass it on to the step manager that would help to execute the planned steps
    """

    system_prompt = SystemMessage(content=prompt)

    response = manager_model.invoke([system_prompt] + state["messages"])

    msg = f"""
    Plan Manager:
    Next Agent: {response.next_agent}
    Explanation: {response.explanation}
    Instructions: {response.instructions}    
    """

    print(msg)

    return Command(
        update={"messages": [AIMessage(content=response.instructions)]},
        goto={response.next_agent}
    )

def perceptor_agent(state: OverallState):
    last_msg = state["messages"][-1].content

    input = {
        "messages": last_msg,
        "destination": state["destination"],
        "object_details": None,
        "next_agent": None
    }

    response = perceptor_subgraph.invoke(input)

    object_details = response["object_details"]

    msg = f"Perceptor: These are the surveyed positional details: {json.dumps(object_details, indent=2)}"
    print(msg)

    return {
        "messages": [AIMessage(content=msg)],
        "object_details": object_details
    }

# parse steps
def parseSteps(steps: list[dict]):
    report = ""
    for step in steps:
        function = step["function"]
        argument = step["argument"]
        explanation = step["explanation"]
        msg = f"""
        ----Step---- 
        Function: {function}
        Argument: {argument}
        Explanation: {explanation}
        """
        report += msg

    return report
    
def planner_agent(state: OverallState):
    last_msg = state["messages"][-1].content

    if not state["object_details"]:
        msg = "No object details found, please retrieve that first"
        print(msg)
        return {
            "messages": [AIMessage(content=msg)]
        }

    input = {
        "messages": last_msg,
        "destination": state["destination"],
        "object_details": state["object_details"],
    }

    response = planner_subgraph.invoke(input)

    agent_msg = response["messages"][-1].content
    steps = response["steps"]

    report = parseSteps(steps)

    msg = f"Planner: {agent_msg}. Steps: {report}"

    return {
        "messages": [AIMessage(content=msg)],
        "steps": steps
    }

# step manager


step_manager_tools = [rotateDrone, moveDroneForward]
step_manager_model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(step_manager_tools)

def StepManager(state: OverallState):
    report = parseSteps(state["steps"])
    prompt = """
    You are the step manager agent in charge of a drone

    End Goal:
    You need to make sure a list of steps is executed fully 

    For step in the list steps is a dictionary as follows:
    - explanation: An explanation for the step
    - function: The name of the function to be called in the step
    - argument: The argument of the function 

    Here are the available tools you have:
    - rotateDrone tool with arguments: `degrees` which will rotate the drone by degrees specified
    - moveDroneForward tool with argument: `distance` to move the drone forward by distance specified
   
    You are to execute the steps from top to bottom in the list
    Here is the current state of steps:
    {report}

    Rules:
    - For every step, you are to:
        - call the respective tool that executes that step's function
    - repeat the process until you have executed all the steps in the list of steps then STOP CALLING TOOLS
    """
    
    system_prompt = SystemMessage(content=prompt)
    cur_steps = SystemMessage(content=f"Here are the current steps: {report}")

    response = step_manager_model.invoke([system_prompt] + state["messages"] + [cur_steps])

    msg = response.content
    if msg:
        print(msg)
    print(f"StepManager: These are the current steps: \n{report}")

    return {
        "messages": [response]
    }


def should_continue_steps(state: OverallState):
    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "step_manager_tools"
    
    return "verifier"


def verify_destination(state: OverallState):

    obj = state["destination"]
    
    ans = checkObject_raw(obj)
    
    if ans == "no":
        msg = f"Seemed to have lost sight of the {obj}, perhaps we would need to relocate the object and plan path again"
        print(msg)
        state["messages"].clear()
        return {
            "message": [AIMessage(content=msg)]
            }

    image = getDroneImage_raw()
    info = calculateMovementsToObjects(image, [state["destination"]])

    if not info or info.get(state["destination"], None) == None:
        msg = f"Seemed to have lost sight of the {obj}, perhaps we would need to relocate the object and plan path again"
        print(msg)
        state["messages"].clear()
        return {"message": [AIMessage(content=msg)]}
    
    distToObject = info[obj][0]["forward_distance_after_yaw_rot"]

    if distToObject < 3:
        msg = "Destination reached!"
        print(msg)
        return {
            "is_found": True,
            "messages": [AIMessage(content=msg)]
            }
    else:
        msg = "Destination not reached! Object is within sight but too far away"
        print(msg)
        return {
            "messages": [AIMessage(content=msg)]
        }
    
def should_end(state: OverallState):
    ans = state["is_found"]

    if ans:
        return "end"
    
    return "plan_manager"

graph = StateGraph(OverallState, input=InterpreterState)
graph.set_entry_point("interpreter")
graph.add_node("interpreter", interpreter)
graph.add_node("plan_manager", PlanManager)

graph.add_node("planner_agent", planner_agent)
graph.add_node("perceptor_agent", perceptor_agent)
graph.add_node("step_manager", StepManager)
graph.add_node("step_manager_tools", ToolNode(step_manager_tools))
graph.add_node("verifier", verify_destination)

graph.add_edge("interpreter", "plan_manager")
graph.add_edge("step_manager_tools", "step_manager")
graph.add_edge("perceptor_agent", "plan_manager")
graph.add_edge("planner_agent", "plan_manager")

graph.add_conditional_edges(
    "verifier",
    should_end,
    {
        "end": END,
        "plan_manager": "plan_manager"
    }
)

graph.add_conditional_edges(
    "step_manager",
    should_continue_steps,
    {
        "step_manager_tools": "step_manager_tools",
        "verifier": "verifier"
    }
)

app = graph.compile()
config = RunnableConfig(recursion_limit=50)

input = {
    "messages": [("user", "Go to the red car")],
    "destination": None
}

app.invoke(input, config=config)


"""


    
def executor_agent(state: ManagerState):
    last_msg = state["messages"][-1].content

    input = {
        "messages": last_msg,
        "destination": state["destination"],
        "object_details": state["object_details"],
    }

    class ExecutorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    step: dict

class ExecutorOutput(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
"""



# main agent for navigation tracking
    # subagent for perception
    # subagent for execution of task
    # subagent for path planning


"""
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    # the latest command of the user
    destination: str | None
    queue: list[dict]
    is_complete: bool
    is_found: bool

# 
class AgentsSelection(BaseModel):
    message_type: Literal["our_finder", "our_pathMaker", "our_executor"] = Field(
        ...,
        description="based on the message, output the next agent that is suitable for the next job"
    )

tools = [rotateDrone, checkObject]

model = ChatOpenAI(model="gpt-4o", api_key=api_key)
finder_model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(tools, parallel_tool_calls=False)
structured_model = ChatOpenAI(model="gpt-4o", api_key=api_key).with_structured_output(AgentsSelection)

# interpreter agent
def interpreter(state: AgentState):
    user_prompt = state["messages"]
    system_prompt = SystemMessage(
        content="You are an interpreter, given a prompt by the user, you need to interprete the destination of the user"
        "Your output should only be (destination) where destination is the destination you identified, no words before that, after that, and no punctuations whatsoever, just the destination"
    )
    response = model.invoke(
        [system_prompt] + user_prompt,
        )
    msg = f"{response.content} is the identified destination"
    print(msg)
    return {
        "messages": [response],
        "destination": response.content,
    }


    
def manager(state: AgentState) -> Command[Literal["our_finder", "our_pathMaker", "our_executor"]]:
    system_prompt = SystemMessage(
        content="You are a navigational assistant. "
        "You have access to a finder agent that would rotate the drone until the drone faces the target destination"
        "You have access to a pathMaker agent that would load the necessary drone movements to the destination (after it is found) into the state queue"
        "You also have a executor agent that would execute the first command in the state queue if the queue is not empty"
        "Your goal is to find the destination using finder, then derive the directions to it using pathMaker, and then execute all movements in the queue using executor"
        "only then would you arrive at your destination"
        "there is no reason to call executor is the state queue is empty"
        "You are to output only from the three choices here: finder, pathMaker or executer, no punctuations or whitespace, just the name of whatever agent you want to call next"
    )

    response = structured_model.invoke([system_prompt] + state["messages"])
    print(f"{response.message_type} has been chosen next")

    return Command(
        update={"messages": [AIMessage(content=response.message_type)]},
        goto={response.message_type}
    )

def finder(state: AgentState):
    system_prompt = SystemMessage(
        content="You are an object finder equipped with tools to use" 
        "You have access to a rotateDrone tool with arguments: `degrees` which will rotate the drone by degrees specified"
        "You have access to a checkObject tool with argument: `object` to see if the object has been found"
        "Keep rotating the drone and checking if the state's destination is found until you find the destination"
        "you MUST actually call these tools"
        "in case the object is out of sight through the process, just repeat the finding steps from the start again"
    )

    response = finder_model.invoke([system_prompt] + state["messages"])
    print(f"Finder Agent: {response.content}")
    
    return {"messages": [response]}

def pathMaker(state: AgentState):
    image = getDroneImage_raw()
    rotation, y_move, forward_move, _ = calculateMovementsToObject(image, state["destination"])

    if rotation == None:
        obj = state["destination"]
        msg = f"Seemed to have lost sight of the {obj}, perhaps we would need to relocate the object and plan path again"
        return {
            "messages": [AIMessage(content=msg)],
            "is_found": False
        }
    
    rotation = (rotation - 5) if rotation > 0 else (rotation + 5)
    
    queue = []

    queue.append({
        "name": "rotateDrone",
        "function": rotateDrone_raw,
        "arg": rotation})
    
    queue.append({
        "name": "moveDroneForward",
        "function": moveDroneForward,
        "arg": forward_move - 0.6
    })

    msg = "Loaded the drone movement functions for the drone into the state queue"
    print(f"pathMaker Agent: {msg}")

    return {
        "messages": [AIMessage(content=msg)],
        "queue": state["queue"] + queue
        }

def executor(state: AgentState):

    if len(state["queue"]) > 0:
        queue = state["queue"]
        latest_cmd = queue.pop(0)
        func = latest_cmd["function"]
        arg = latest_cmd["arg"]
        name = latest_cmd["name"]
        response = func(arg)


        return {
            "messages": [AIMessage(content=response)],
            "queue": queue
        }
    
    msg = "queue is empty?"
    print(f"Executor Agent: {msg}")
    return {
        "messages": [AIMessage(content=msg)]
    }

def verify_destination(state: AgentState):

    if len(state["queue"]) > 0:
        print("Still have commands left in the queue")
        return state
    
    image = getDroneImage_raw()
    _, _, _, dist = calculateMovementsToObject(image, state["destination"])
    if dist == None:
        obj = state["destination"]
        msg = f"Seemed to have lost sight of the {obj}, perhaps we would need to relocate the object and plan path again"
        print(msg)
        return {"message": [AIMessage(content=msg)],
                "is_found": False}


    print(f"distance from object is now {dist} m")
    if dist < 3:
        print("Destination reached")
        return {"is_complete": True, "messages": [AIMessage(content="Destination reached")]}
    else:
        print("Destination not reached")
        return state

# conditional edge
def should_continue_from_finder(state: AgentState):
    if state["is_found"]:
        dest = state["destination"]
        print(f"Seems like we found the {dest}, going back to agent")
        return "agent"
    
    return "continue"

def check_completion(state: AgentState):
    if state["is_complete"]:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.set_entry_point("our_interpreter")

graph.add_node("our_interpreter", interpreter)
graph.add_node("our_agent", manager)

graph.add_node("our_finder", finder)
graph.add_node("finder_tools", ToolNode(tools))

graph.add_node("our_executor", executor)
graph.add_node("our_verifier", verify_destination)

graph.add_node("our_pathMaker", pathMaker)


graph.add_edge("our_interpreter", "our_agent")
graph.add_edge("our_pathMaker", "our_agent")
graph.add_edge("our_executor", "our_verifier")
graph.add_edge("finder_tools", "our_finder")

graph.add_conditional_edges(
    "our_verifier",
    check_completion,
    {
        "continue": "our_agent",
        "end": END
    }
)

graph.add_conditional_edges(
    "our_finder",
    should_continue_from_finder,
    {
        "continue": "finder_tools",
        "agent": "our_agent"
    }
)


app = graph.compile()
config = RunnableConfig(recursion_limit=50)


inputs = {"messages": [("user", "Go to the red car")],
            "destination": None,
            "queue": [],
            "is_complete": False,
            "is_found": False}


def print_stream(stream):
  for s in stream:
    message = s["messages"][-1]
    if isinstance(message, tuple):
      print(message)
    else:
      message.pretty_print()

app.invoke(inputs, config=config)

"""





























# commands for the agents

# interpreter
# manager
# findobject
    # get image
    # rotate



# plan path
# execute plan


# camera_command()
# move_command
# calculateMovementsToObject(response, "red car")
    # returns rotation, y_move, forward move












































        