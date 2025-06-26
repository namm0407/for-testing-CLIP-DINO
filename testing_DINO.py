#sucessfully detected the cat, but it can't detect the dog

import torch
import cv2
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt

#Load the pre-trained Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

#Load an image
#image_url = "https://example.com/sample_image.jpg"  # Replace with a valid image URL
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open("cat.png").convert("RGB")

#Define the text prompt
text_prompt = "a dog, a cat, a red car"  # Objects to detect

#Preprocess the inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

#Run inference
with torch.no_grad():
    outputs = model(**inputs)

#Process the outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.3,  # Confidence threshold for boxes
    text_threshold=0.3,  # Confidence threshold for text
    target_sizes=[image.size[::-1]]  # (height, width)
)[0]

#Visualize the results
print("Detected objects:")
for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    print(f"- {label}: Confidence = {score:.2f}, Box = {box}")

image_np = np.array(image)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    if score > 0.3:  # Filter low-confidence detections
        box = [int(b) for b in box]
        cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            image_cv,
            f"{label} {score:.2f}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

#Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#Save the output image (optional)
#cv2.imwrite("cat.png", image_cv)
