import sys
from pathlib import Path
import cv2
import torch
import pillow_heif
from PIL import Image
from ultralytics import YOLO
import pillow_heif
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from segment_anything import SamPredictor, sam_model_registry
import math
from transformers import SamModel, SamProcessor

depth_anything_path = Path("../Depth-Anything-V2/metric_depth")
sys.path.append(str(depth_anything_path))

from depth_anything_v2.dpt import DepthAnythingV2

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
CLASSES = ["table", "sofa"]
yolo_model.set_classes(CLASSES)

# edge cases "door", "cane"
# compare with the current approach

# agentic framework
    # find errors 
    # 

# position paper
    # read other papers
    # try a bit on the current drone
    # 'someone said it is better on other cases, we can try it out on


# write-up on depth thingy
    # code it out by today
    # 


# SEGMENT-ANYTHING MODEL
sam_checkpoint = "sam_vit_b_01ec64.pth"
model = "vit_b"
sam = sam_model_registry[model](checkpoint=f"../Segment-Anything/checkpoints/{sam_checkpoint}")
sam.to(device="cpu")
predictor = SamPredictor(sam)

# SAM 2
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam2_model = SamModel.from_pretrained("facebook/sam-vit-large")

sam2_model.eval()





def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



# model results generators
def depth_anything(model, array):
    start = time.time()
    results = model.infer_image(array)
    end = time.time()
    latency_ms = (end - start) * 1000 
    return results, latency_ms

def yolo_world(model, array):
    start = time.time()
    results = model.predict(array)
    end = time.time()
    latency_ms = (end - start) * 1000 
    return results, latency_ms

def segment_anything(predictor, box, multimask_output=False, point_coords=None, point_labels=None):
    start = time.time()
    masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output
            )
    end = time.time()
    latency_ms = (end - start) * 1000 
    return masks, scores, logits, latency_ms

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

    
# 2102
# 2106

# loop
PIC_DIR = Path("../pictures")
pictures = list(PIC_DIR.glob("*.jpg"))

output_dict = []
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720
horizontal_fov = 70
vertical_fov = 54

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



annotations = []

pictures = [Path(f"{PIC_DIR}/test4.jpg")]


for i in range(len(pictures)):
    start = time.time()
    picture = pictures[i]
    img = Image.open(picture).convert("RGB")
    array = np.array(img)


    depth, depth_latency = depth_anything(depth_model, array)
    results, yolo_latency = yolo_world(yolo_model, array)

    print(f"---------------------PICTURE {i}---------------------")
    print(f"Depth Anything Latency: {depth_latency:.2f} ms")
    print(f"Yolo World Latency: {yolo_latency:.2f} ms")

    # predictor.set_image(array)
    for result in results:
        xyxy = result.boxes.xyxy
        boxes = result.boxes
        names = result.names
        cls = result.boxes.cls
        masks_lst = []
        for j in range(len(xyxy)):
            box_xyxy = xyxy[j].cpu().numpy()
            

            mask, sam_latency = sam_2(array, box_xyxy.tolist())
            # transformed_box = predictor.transform.apply_boxes(np.array([box_xyxy]), array.shape[:2])
            # masks, scores, logits, sam_latency = segment_anything(predictor, box=transformed_box[0], multimask_output=True)
            print(f"mask latency: {sam_latency:.2f} ms")

            depth_values = depth[mask]
            depth_min = np.min(depth_values[depth_values > 0])


            name = names[int(cls[j])]
            print(f"--------ITEM: {name}--------")
            print(f"estimated depth: {depth_min:.2f} meters")

            # find the centroid of the object
            ys, xs = np.where(mask)
            centroid_y = ys.mean()
            centroid_x = xs.mean()
            x, y, z, yaw_rot, pitch_rot, diag_perpen_dist  = getDirections((centroid_y,centroid_x), estimated_depth=depth_min)
            print(f"Drone needs to rotate {yaw_rot} degrees along x-axis, {y} meters along y-axis, and {diag_perpen_dist} meters forward")

            masks_lst.append(mask)

        annotated = result.plot()
        annotations.append((annotated, masks_lst))
    
    end = time.time()
    latency = (end - start) * 1000
    print(f"Total time for picture {i}: {latency:.2f} ms")


n = len(annotations)
cols = min(n, 3)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

if n == 1:
    # make iterable
    annotation, masks = annotations[0]
    axes.imshow(annotation)

    for mask in masks:
        show_mask(mask, axes, True)

    axes.axis("off")
else:
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            annotation, masks = annotations[idx]
            ax.imshow(annotation)

            for mask in masks:
                
                show_mask(mask, ax, True)
                """
                colored_mask = mask.astype(float)
                ax.imshow(np.ma.masked_where(~mask, colored_mask), cmap='jet', alpha=0.5)
                """

            ax.axis("off")
        else:
            ax.axis("off")

plt.tight_layout()
plt.show()








"""
output_dict.append((depth, results, depth_latency, yolo_latency))

annotations = []


for i, (depth, results, depth_latency, yolo_latency) in enumerate(output_dict):
    print(f"---------------------PICTURE {i}---------------------")
    print(f"Depth Anything Latency: {depth_latency:.2f} ms")
    print(f"Yolo World Latency: {yolo_latency:.2f} ms")
    for result in results:
        xyxy = result.boxes.xyxy
        for i in range(len(xyxy)):
            cur = xyxy[i]
            x1, y1, x2, y2 = cur
            x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
            width = x2 - x1
            height = y2 - y1
            mid_x = round(x1 + width / 2)
            mid_y = round(y1 + height / 2)
            estimated_depth = depth[mid_y][mid_x].min()
            label = result.names[int(result.boxes.cls[i])]
            print(f"--------ITEM: {label}--------")
            print(f"top left: {x1}, {y1}")
            print(f"bottom right: {x2}, {y2}")
            print(f"estimated depth: {estimated_depth:.2f} meters")
        annotated = result.plot()
        annotations.append(annotated)

n = len(annotations)
cols = min(n, 3)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

if n == 1:
    axes = [axes]  # make iterable

for idx, ax in enumerate(axes.flat):
    if idx < n:
        ax.imshow(annotations[idx])
        ax.axis("off")
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()



"""