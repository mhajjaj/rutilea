import os
from predictor import Predictor as predictor

import cv2 
from imutils import perspective
import numpy as np 
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from imutils import perspective

import cv2 
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# HOME = os.getcwd()
HOME = f"C:\\Users\\mhajj\\Documents\\RUTILEA\\Signateprojects"

images_path = f"{HOME}\\dataset\\train\\images"
annotation_path = f"{HOME}\\dataset\\train\\annotation"

print("Number of Training images are", len(os.listdir(images_path)))

count = 0
annotations = []  # Create a list to store all annotations

for i in tqdm(os.listdir(images_path)):
    img_path = os.path.join(images_path, i)
    annotation_file_path = os.path.join(annotation_path, i.split(".")[0] + ".json")
    
    try:
        with open(annotation_file_path, "r") as json_file:
            annotation_data = json.load(json_file)
            
            # Extract the data you need from the loaded JSON
            nutrition_facts = annotation_data.get("nutrition_facts")
            calory = annotation_data.get("calory")

            # Create the annotation dictionary
            annotation = {
                "nutrition_facts": nutrition_facts,
                "calory": calory
            }

            annotations.append(annotation)
            
    except FileNotFoundError:
        continue

    count += 1

# Write all annotations to a JSON file
with open("train_annotations.json", "w") as json_file:
    json.dump(annotations, json_file, indent=4)

print("Total number of Annotations created for Training are", count)

count = 0
annotations = []  # Create a list to store all annotations

for i in tqdm(os.listdir(images_path)):
    img_path = os.path.join(images_path, i)
    annotation_file_path = os.path.join(annotation_path, i.split(".")[0] + ".json")
    
    try:
        with open(annotation_file_path, "r") as json_file:
            annotation_data = json.load(json_file)

            # Extract the data you need from the loaded JSON
            nutrition_facts = annotation_data.get("nutrition_facts")
            calory = annotation_data.get("calory")

            # Create the annotation dictionary
            annotation = {
                "nutrition_facts": nutrition_facts,
                "calory": calory
            }

            annotations.append(annotation)
            
    except FileNotFoundError:
        continue

    count += 1

# Write all annotations to a JSON file
with open("test_annotations.json", "w") as json_file:
    json.dump(annotations, json_file, indent=4)

print("Total number of Annotations created for Test/Eval are", count)

# Initialize PaddleOCR
custom_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    show_log=False
)

custom_ocr = PaddleOCR(
    use_angle_cls=True,
    rec_model_dir='..\\model\\PaddleOCR\\inference\\en_PP-OCRv3_rec',
    det_model_dir='..\\model\\PaddleOCR\\output\\det_db_inference',
    # rec_char_dict_path='..\\model\\PaddleOCR\\ppocr\\utils\\dict90.txt',
    lang='en',  
    use_gpu=True,
    show_log=True
)

# Iterate through the images in the folder
for image_filename in os.listdir(images_path):
    if image_filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(images_path, image_filename)
        
        # Perform OCR on the current image
        results = custom_ocr.ocr(img_path)

        if not results:
            print(f"No text found in {img_path}.")
        else:
            # Process the OCR results
            print(f"Text found in {img_path}:")
            for result in results[0]:
                _, (text, confidence) = result
                print(f'Text: {text}, Confidence: {confidence}')

            # Display or save the result image with bounding boxes
            image = Image.open(img_path).convert('RGB')
            boxes = [detection[0] for line in results for detection in line]
            txts = [detection[1][0] for line in results for detection in line]
            scores = [detection[1][1] for line in results for detection in line]

            im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
            im_show = Image.fromarray(im_show)
            # im_show.show()
            im_show.save(f'..\\dataset\\output\\{image_filename}')