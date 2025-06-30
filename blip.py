# For testing if BLIP is setted up in the local machine

from lavis.models import load_model_and_preprocess
import torch

# Load BLIP-2 model (example: pretrained OPT-2.7B)
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("BLIP-2 model loaded successfully!")
