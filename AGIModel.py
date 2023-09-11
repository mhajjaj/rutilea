import os
import torch
from super_gradients.training import models

class config:
    PATH = f"{os.getcwd()}\AGIExperiment\AGIModel.pt"
    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
    MODEL_ARCH = 'yolo_nas_l' 
    PRETRAINED_WEIGHTS = 'coco'

model = models.get(config.MODEL_ARCH, config.PRETRAINED_WEIGHTS).to(config.DEVICE)

checkpoint = torch.load(config.PATH)
# model.load_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(torch.load(config.PATH))
model.eval()