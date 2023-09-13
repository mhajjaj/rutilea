import os
HOME = os.getcwd()
print("HOME:", HOME)

import supervision as sv
print(sv.__version__)

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from groundingdino.util.inference import load_model, load_image, predict, annotate
model = load_model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_NAME)

IMAGE_NAME = "year=2023-month=01-day=09-10_42_34-OK-2_1.png" 
IMAGE_DIR = f"{HOME}\\GearInspection-Dataset2\\valid\\images\\"
IMAGE_PATH = os.path.join(HOME, IMAGE_DIR, IMAGE_NAME)

print("IMAGE_PATH:", IMAGE_PATH)

TEXT_PROMPT = "kizu"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
sv.plot_image(annotated_frame, (16, 16))