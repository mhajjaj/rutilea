import os
from predictor import Predictor as predictor

import cv2 
import os 
from imutils import perspective
import numpy as np 
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from imutils import perspective

HOME = os.getcwd()
print(HOME)
print("test")

images_path = os.listdir(f"{HOME}\\..\\dataset\\train")

print("Number of Training images are", len(images_path))