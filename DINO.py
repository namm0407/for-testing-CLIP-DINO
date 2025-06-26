#sucessfully detected objects (have a green box boxxing the detected object with a small name on the top)
#updated to show the name of the detected object in the terminal.

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the pre-trained Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# Load an image (any image) 
# will only show the last one
image = Image.open("cat.png").convert("RGB")
image = Image.open("dogs.jpg").convert("RGB")

# Define the text prompt with proper class separation
text_prompt = "dog . cat . red car"  # Use periods to separate classes

# Preprocess the inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Process the outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,  # Increased threshold for better filtering
    text_threshold=0.4,  # Increased threshold for better text matching
    target_sizes=[image.size[::-1]]  # (height, width)
)[0]

# Visualize the results
print("Detected objects:")
for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):  # Use text_labels
    print(f"- {label}: Confidence = {score:.2f}, Box = {box}")

image_np = np.array(image)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):  # Use text_labels
    if score > 0.4:  # Filter low-confidence detections
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

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Step 9: Save the output image (optional)
# cv2.imwrite("output_cat.png", image_cv)
