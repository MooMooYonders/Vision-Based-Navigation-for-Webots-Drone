# Vision-Based-Navigation-for-Webots-Drone

**Exploring the Utility of Agentic LLM-Based Systems in Supporting Goal-Driven Navigation for UAVs for BLVs**

**Introduction**

<p style="text-align: justify;">
Globally, at least 2.2 billion people have blindness or low vision (BLV) (WHO, 2023). Such a figure positions accessibility research as an increasingly important and topical research area for the field of Human-Computer Interaction (HCI) (Mack, 2022). Multitudinous assistive technologies have been presented as a viable means to help the BLV in their everyday tasks. Among these, unmanned aerial vehicles (UAV), or drones, have emerged as a promising modality for assisting BLV individuals with navigation. (Aibid et al., 2024)

Recent advancements in Artificial Intelligence (AI), particularly the development of Large Language Models (LLMs), have enabled new and advanced forms of semantic understanding and reasoning (Hagos et al., 2024). One such advancement is the emergence of agentic systems.  This agentic concept in LLM-based applications refers to developing Artificial Intelligence (AI) systems that can act autonomously, make decisions and perform tasks with minimal human intervention. (Pankaj, 2024). This opens up novel possibilities for enhancing the interaction between BLV users and assistive technologies through natural language, as well as creating independent systems run entirely by AI.

This work presents an early-stage investigation into the use of agentic LLMs for closed-loop, vision-based navigational support in drones. We describe our prototype architecture, analyse its design trade-offs, and assess its strengths and limitations in the context of potential applications for BLV community.
</p>

**Background**

<p style="text-align: justify;">
Prior research has explored the use of drones to assist visually impaired individuals with a primary focus on vision-based tasks. For instance, in the Flying Guide Dog, Tan et al. (2021) leveraged the predictive capabilities of segmentation models to help discover walkable areas and identify pedestrian lights in order to assist BLV users in walking outdoors.
On the other hand, Zhang et al.'s (2025) paper involved Vision Language Models which employed advanced reasoning on complex semantic signs, translating these into mobility information to the user.

However, these systems often relied on hardcoded destinations. There was also limited flexibility in interactions with the users. The potential of LLMs to act as high-level cognitive agents that interpret commands and plan actions remains underexplored.
In order to address this gap, we investigate whether agentic LLM-based systems can serve as a more adaptable and semantically rich alternative for goal-driven drone navigation, particularly for BLV users.
</p>

**Methods**

<p style="text-align: justify;">
We adopt a supervisor-agent architecture as shown in Fig. 1. The Interpreter serves as the graph’s initial entry point, responsible for understanding and extracting the user's intent as well as identifying the target destination from natural language queries. Once the objective is clear, the Plan Manager takes over, orchestrating the overall strategy required to achieve the goal. It accomplishes this by coordinating with two specialised sub-agents: the Perceptor Agent and the Planner Agent. The Perceptor Agent is tasked with locating the target destination and identifying any obstacles or surrounding elements in the environment. It supplies the Plan Manager with positional and depth data for the identified objects. Based on these spatial inputs, the Planner Agent then formulates a series of steps that outlines how the drone should navigate toward its target.
After the plan is constructed, it is handed off to the Step Manager, which is responsible for executing the strategy incrementally. This is accomplished through direct control of the drone using commands such as rotateDrone and moveDroneForward. Finally, the Verifier Agent ensures the success of the mission by checking whether the drone has arrived at the intended destination. If the destination has been reached, the multi-agent system is exited. However, if the drone has not yet reached its goal, the Verifier re-engages the Plan Manager to initiate a new round of replanning. This helps to ensure an adaptive and robust system.
</p>

<p align="center">
<img width="432" height="380" alt="image2" src="https://github.com/user-attachments/assets/bff6c40e-af7e-41f4-8d86-6db5858aca92" />
</p>

<p align="center">
Fig 1. Overall Framework for Agentic Model
</p>

**Simulation Setup**

<p style="text-align: justify;">
We implemented and tested our system in the Webots simulator due to its wide usage by industry researchers in robotics and AI (Webots, n.d.). The Mavic-2-Pro drone environment was used. The drone was equipped only with a forward-facing camera, to match the capabilities of the real DJI Tello drone we plan to test with BLV participants.

Given that our system relies on vision-based inputs, accurate depth estimation was critical. We integrated three computer vision models - YOLO-World, Depth Anything V2 and Segment Anything. 
Firstly, YOLO-World enables object detection based on natural language prompts, allowing users to specify objects of interest using textual descriptions rather than fixed class labels. In their research paper, Cheng et al. (2024) present an architecture that supports open-vocabulary detection through a vision-language pipeline. 
Unlike traditional YOLO models that output class probabilities over a fixed set of categories, YOLO-World introduces a Vision-Language PAN (RepVL-PAN) that fuses multi-scale image features with text information. Text-guided CSP Layers (T-CSPLayer) and Image-Pooling Attention are integrated within the Vision-Language PAN, enabling richer interactions between the visual features and the input text embeddings. These modules allow the model to generate both text-aware visual features and image-aware text embeddings. This enhances the model’s ability to understand what the user is asking for. From these fused features, the model predicts bounding boxes as usual, but instead of class scores, it outputs a feature embedding per box. These embeddings represent what the model thinks the object could be. During inference, these are compared with the embeddings of the text prompts via cosine similarity, and the highest similarity matches are retained as detections. During training, the model compares the predicted boxes against ground truth boxes and for each matched pair, it also aligns the predicted feature embedding with the ground truth text label’s embedding using a contrastive loss. This lets the model learn not just where target objects are, but also how to match them to textual descriptions.
This function complements well with our  LLM-focused multi-agent system. Add one more sentence here to explain why  The rich and descriptive text prompts generated by LLMs can be directly inputted into YOLO-World to perform targeted visual detection based on the user’s request. This allows for seamless coordination between language and vision. 

Secondly, Depth Anything V2 is used to generate dense depth maps from standard RGB images. According to Yang et al. (2024), Depth Anything V2’s ability to output depth maps from images stems from using a pseudo-supervised learning approach. Instead of relying on labeled depth data, it uses a pre-trained “teacher” model to generate depth predictions, which serve as pseudo-ground truth. A student model is then trained to replicate these predictions, with the difference between the student’s and teacher’s outputs used as the loss function during training.
Semantic-assisted perception is incorporated to further improve the model’s depth estimation. While processing the image, the student model encodes it into internal feature vectors. These are compared to features extracted from a frozen DINOv2 encoder, which is known for capturing rich semantic information about objects and scenes. A feature alignment loss based on cosine similarity encourages the student’s internal representations to match DINOv2’s. This guides the student model to learn depth in a way that is semantically meaningful which results in more accurate and robust depth predictions. By introducing semantic understanding in the student model, this enhances its ability to generalise to diverse, real-world images outside of its narrow, specialised datasets. This function makes Depth Anything an ideal model for our use case.
By extracting the camera images from the drone and putting the images through the model, this provides crucial spatial information about the surrounding environment without requiring specialised depth sensors. 

Lastly, to enable fine-grained object-level analysis, the Segment Anything Model (SAM) is incorporated to extract pixel-accurate segmentation masks within detected bounding boxes. 

Kirillov et al. (2023) elaborates how their SAM pipeline first processes the input image using a Vision Transformer (ViT) backbone, which converts the image into high-dimensional feature embeddings that capture global contextual information. To segment a target object, a user-provided prompt such as a coarse mask, a point or a bounding box is provided. This prompt will be encoded using a lightweight prompt encoder. By fusing the encoded prompt with the image features by the Vision Transformer, the model then generates one or more binary masks corresponding to the target region.

These masks delineate which pixels belong to the object of interest, beyond the coarse localisation of bounding boxes. This capability allows for more precise focus on specific objects, making it possible to perform targeted depth queries for them with greater accuracy.
</p>

**Key Function: Depth Estimation and Movement Calculation**

To extract actionable depth and positioning data for navigation, we used the following function:

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

<p style="text-align: justify;">
This function takes in a list of identified objects by the LLM. It first applies YOLO-World to the image from the drone’s camera to identify the bounding boxes for each of those target objects. Then, it utilises Segment Anything to isolate those objects even further and later uses the depth map generated from Depth-Anything to estimate the straight-line distance from drone to object.
Below is an illustration on how the function works.
</p>

<p align="center">
<img width="1536" height="1024" alt="image3" src="https://github.com/user-attachments/assets/7797afd6-cf3a-49b9-8980-e5a60a978a5a" />
</p>

<p align="center">
Fig 2. Illustration on how the three models are used to derive depth estimation
</p>

<p style="text-align: justify;">
Referring to Fig 3, using the field of view of the camera, the horizontal and vertical angles can be derived which, together with the depth estimation, allows us to attain the relative coordinates of the object relative to the drone’s position. The perceptor agent uses this output to inform the planner agent, allowing the agent to better plan the navigational steps for the drone.
</p>

<p align="center">
<img width="1536" height="1024" alt="image1" src="https://github.com/user-attachments/assets/b03a19e3-f1a1-427e-87fe-029fed5433bc" />
</p>

<p align="center">
Fig 3. Illustration showing how relative coordinates of object is derived from straight-line distance
</p>

**Results**

<p style="text-align: justify;">
Our prototype demonstrated core functionality within a simulated environment. The drone was able to successfully identify target objects and the various agents were able to operate autonomously by selecting and invoking the appropriate tools and sub-agents to accomplish their respective objectives. These findings validated the system’s modular design and basic end-to-end workflow.
However, several limitations were highlighted during testing. Latency was a key issue. The processing time required by certain models introduced delays. As the drone’s position could shift during these periods of delay, this led to navigational errors. Moreover, system performance was heavily dependent on the accuracy of object detection and depth estimation. To elucidate, while the YOLO-World model reliably detected common objects such as "red car" or "bottle," it struggled with less familiar items like walking sticks or irregularly-shaped furniture. This demonstrates the need for further fine-tuning and dataset expansion to enhance the model’s ability to recognize a wider range of objects.
Conclusion
This early-stage prototype demonstrates the potential of agentic LLM frameworks for assistive UAV navigation. By combining LLMs with CV models in a modular agent-based architecture, we enable semantically rich, flexible control loops.

While our current implementation was limited to simulation, future work will focus on reducing latency through model optimization and hardware acceleration, real-world testing with BLV participants and improving robustness in dynamic, cluttered environments.

Ultimately, we see this work as an exploratory step towards intelligent, language-driven assistive navigation systems for the visually impaired.
</p>

Demo:

https://drive.google.com/file/d/14-FcjCAc5GNtgB7o_C8i_8lgkEqtEAR-/view?usp=sharing

References:

Abidi, M. H., Siddiquee, A. N., Alkhalefah, H., & Srivastava, V. (2024). A comprehensive review of navigation systems for visually impaired individuals. Heliyon, 10(11), e31825. https://doi.org/10.1016/j.heliyon.2024.e31825

Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024, January 30). YOLO-World: Real-Time Open-Vocabulary Object Detection. arXiv.org. https://arxiv.org/abs/2401.17270

Hagos, D. H., Battle, R. & Rawat, D. B. (2024, August 2). Recent advances in generative AI and large language models: current status, challenges, and perspectives.
https://arxiv.org/html/2407.14962v3

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W., Dollár, P., & Girshick, R. (2023, April 5). Segment anything. arXiv.org. https://arxiv.org/abs/2304.02643

Mack, K. A. (2022, January 6). What do we mean by “Accessibility research”? - HCI & Design at UW - Medium. Medium.
https://medium.com/hci-design-at-uw/what-do-we-mean-by-accessibility-research-6b6560620e6d

Pankaj. (2024, November 20). The Agentic concept in LLM-based application development. Medium.
https://medium.com/@pankaj_pandey/the-agentic-concept-in-llm-based-application-development-48beea5cc00d

Tan, H., Chen, C., Luo, X., Zhang, J., Seibold, C., Yang, K., & Stiefelhagen, R. (2021, August 16). Flying Guide Dog: Walkable path discovery for the visually impaired utilizing drones and transformer-based semantic segmentation. arXiv.org. https://arxiv.org/abs/2108.07007

Webots. (n.d.). Webots Is a Free and Open-source 3D Robot Simulator Used in Industry, Education, and Research - Third-Party Products & Services - MATLAB & Simulink. https://ww2.mathworks.cn/en/products/connections/product_detail/webots.html

World Health Organization (2023, August 10). Blindness and vision impairment. https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment

Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024, June 13). Depth Anything v2. arXiv.org. https://arxiv.org/abs/2406.09414

Zhang, Z., Hu, C., Lye, S. & Chen, L. (2025, January 21) A VLM-Drone System for Indoor Navigation Assistance with Semantic Reasoning for the Visually Impaired. (2025, January 21). IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/document/10871009

