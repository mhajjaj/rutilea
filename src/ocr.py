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

custom_ocr = PaddleOCR(
    use_angle_cls=True,
    # rec_model_dir='..\\PaddleOCR\\inference\\en_PP-OCRv3_rec',
    # det_model_dir='..\\PaddleOCR\\output\\det_db_inference',
    # rec_char_dict_path='..\\PaddleOCR\\ppocr\\utils\\dict90.txt',
    lang='en',  
    use_gpu=True,
    show_log=False
)

results = custom_ocr.ocr(img_path)

img = cv2.imread(img_path)
plt.imshow(img)

results = custom_ocr.ocr(img)

if not results:
    print("No text found in the image.")
else:
    # Process the OCR results
    for result in results[0]:
        _, (text, confidence) = result
        print(f'Text: {text}, Confidence: {confidence}')

image = Image.open(img_path).convert('RGB')
boxes = [detection[0] for line in results for detection in line]
txts = [detection[1][0] for line in results for detection in line]
scores = [detection[1][1] for line in results for detection in line]

im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result_ocr.jpg')