import torch
from ultralytics import NAS

print(torch.cuda.is_available())

#model = NAS('yolo_nas_s.pt')
#model.info()
#results = model.val(data='coco8.yaml')
#results = model('mhajjaj\predictImgs\Scratches\year=2023-month=04-day=01-04_36_34-NG-2_1.png')


import os
import cv2

file_path = 'predictImgs\\Scratches\\year=2023-month=04-day=01-04_36_34-NG-2_1.png'

if os.path.exists(file_path):
    # File exists; you can proceed with your operations.
    print(f"OK")
    from super_gradients.training import models
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model=models.get('yolo_nas_s',pretrained_weights="coco").to(device)
    #out=model.predict(file_path)
    
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out=model.predict(img)
    out.show()
else:
    print(f"File not found: {file_path}")
