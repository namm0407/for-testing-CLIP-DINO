from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load the CLIP-ViT-L-336px model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Load an image
image = Image.open("cat.png")  # Replace with a path to an image
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)

# Run inference
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

#print(probs) 

labels = ["a cat", "a dog"]  # Match the text prompts used above
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.4f}")

# Identify the most likely label
max_prob_index = probs.argmax().item()
predicted_label = labels[max_prob_index]
print(f"The image is most likely: {predicted_label}")
