import numpy as np
import imageio.v2 as imageio
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
import re


def load_image_files(directory_path):
    """Load all PNG image files from the specified directory."""
    return list(Path(directory_path).glob("*.png"))


def get_split_point(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded_img = cv2.erode(img, kernel, iterations=1)
    profile_y = np.mean(eroded_img, axis=1)
    # Use indexing to extract the first dimension to make it 1-D
    profile_y = profile_y[:, 0]
    y_split, _ = find_peaks(-profile_y, distance=300)

    ### 

    edges = cv2.Canny(np.array(img), 50, 80)
    profile_x = np.mean(edges, axis=0)
    x_right = np.where(profile_x > 5)[0][0]
    x_left = np.where(profile_x[0:img.shape[1] - 100] > 10)[0][-1]

    return y_split, x_right, x_left


def image_split_save(img, y_split, x_left, x_right, n, original_img):
    img = img[:, x_left:x_right + 1]

    for i in range(len(y_split)-1):
        crop_img = img[y_split[i]:y_split[i + 1], ]

        imageio.imwrite(f"GearInspection-Dataset3\\CategoryNG\\ClassAll\\train_split\\{original_img}_{n:03}.png", crop_img)
        n += 1

    return n

def image_rename(img_path):
    match = re.search(r'file_(.*?)\.png', str(img_path))
    if match:
        desired_string = match.group(1)
        print(desired_string)
        return desired_string
        # exit()

def main(input_directory):
    n = 0
    input_directory_path = Path(input_directory)
    for img_path in load_image_files(input_directory_path):
        img = imageio.imread(img_path)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y_split, x_left, x_right = get_split_point(img)

        if not (683 < x_right - x_left < 695):
            print("横幅の異常", x_right - x_left)
            continue

        for i in range(len(y_split)-1):
            if not (308 < y_split[i + 1] - y_split[i] < 320):
                print("縦幅の異常", y_split[i + 1] - y_split[i])
                continue

        n = image_split_save(img, y_split, x_left, x_right, n, image_rename(img_path))

main("GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\images")