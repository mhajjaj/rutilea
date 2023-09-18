from super_gradients.training import models
import os

HOME = os.getcwd()
MODEL_PATH = f'{HOME}\\checkpoint\\AGI-Dataset\\AGIExperiment\\average_model.pth'
NUM_CLASSES = 8

img_path = f'{HOME}\GearInspection-Dataset\predict\year=2023-month=04-day=27-00_16_34-NG-2_10.png'
best_model = models.get('yolo_nas_l', num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)
best_model.predict(img_path, conf=0.25).show()