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
text_prompt = "dog . cat . red car"

# Preprocess the inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Process the outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.4,
    target_sizes=[image.size[::-1]] # (height, width)
)[0]

# Visualize the results
print("Detected objects:")
for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
    print(f"- {label}: Confidence = {score:.2f}, Box = {box}")

image_np = np.array(image)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
image_height, image_width = image_cv.shape[:2]

for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
    if score > 0.4:
        box = [int(b) for b in box]
        x1, y1, x2, y2 = box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate text size and position
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Place text inside the box, near the top
        text_x = x1 + 5
        text_y = y1 + text_size[1] + 5  # 5 pixels from top of box
        
        # Ensure text stays within image bounds
        text_y = max(text_y, text_size[1] + 5)  # Ensure text doesn't go above image
        text_y = min(text_y, image_height - 5)  # Ensure text doesn't go below image
        
        # Add a semi-transparent black rectangle as text background
        bg_x1 = text_x - 2
        bg_y1 = text_y - text_size[1] - 2
        bg_x2 = text_x + text_size[0] + 2
        bg_y2 = text_y + 2
        overlay = image_cv.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0, image_cv)
        
        # Draw text
        cv2.putText(
            image_cv,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Save the output image (optional)
# cv2.imwrite("output_cat.png", image_cv)
