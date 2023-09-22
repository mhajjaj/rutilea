import os
import numpy as np
import imageio.v2 as imageio
import cv2
from scipy.signal import find_peaks
from pathlib import Path
import re

distance = 300

def load_image_files(directory_path):
    """Load all PNG image files from the specified directory."""
    return list(Path(directory_path).glob("*.png"))

def get_annotation_location(image_path):
    image_name = image_path.stem
    label_dir = os.path.join(image_path.parent.parent, "labels")  
    label_path = os.path.join(label_dir, f"{image_name}.txt")
    return label_path 

def get_split_point(img, distance_val):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded_img = cv2.erode(img, kernel, iterations=1)
    profile_y = np.mean(eroded_img, axis=1)
    # Use indexing to extract the first dimension to make it 1-D
    # profile_y = profile_y[:, 0]
    y_split, _ = find_peaks(-profile_y, distance = distance_val)

    edges = cv2.Canny(np.array(img), 50, 80)
    profile_x = np.mean(edges, axis=0)
    x_right = np.where(profile_x > 5)[0][0]
    x_left = np.where(profile_x[0:img.shape[1] - 100] > 10)[0][-1]

    return y_split, x_right, x_left

def parse_and_decode_annotation(annotation_location):
    with open(annotation_location, "r") as file:
        content = file.read()

    values = content.split()

    # Extract class ID and convert it to an integer
    class_id = int(values[0])

    # Extract other values and convert them to floats
    x_center = float(values[1])
    y_center = float(values[2])
    width = float(values[3])
    height = float(values[4])

    return class_id, x_center, y_center, width, height

def convert_annotation_to_pixels(annotation, image_width, image_height):
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = annotation # map(float, annotation.split())

    # Convert normalized coordinates to pixel values
    x_center_px = int(x_center_norm * image_width)
    y_center_px = int(y_center_norm * image_height)
    width_px = int(width_norm * image_width)
    height_px = int(height_norm * image_height)

    # Calculate bounding box coordinates
    # This part is not used at this moment
    x_left_px = x_center_px - (width_px // 2)
    x_right_px = x_center_px + (width_px // 2)
    y_top_px = y_center_px - (height_px // 2)
    y_bottom_px = y_center_px + (height_px // 2)

    return class_id, x_center_px, y_center_px, width_px, height_px

def return_y_center(annotation, image_height):
    return int(annotation[2] * image_height)

def update_y_center(y_center, image_height):
    return float(y_center / image_height)

def update_annotation(image_file_name, new_annotation_location, annotation, image_height, y_center):
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = annotation

    new_y_center_norm = float(y_center / image_height)

    # Get the base name of the image file without the extension
    image_file_name = os.path.splitext(os.path.basename(image_file_name))[0]
    
    # Create the path for the updated annotation file inside the 'labels_adjusted' directory
    new_annotation_location = os.path.join(new_annotation_location, f"{image_file_name}.txt")

    with open(new_annotation_location, "w") as file:
        file.write(f"{class_id} {x_center_norm} {new_y_center_norm} {width_norm} {height_norm}")

    return new_y_center_norm

def convert_pixels_to_annotation(class_id, x_left_px, y_top_px, x_right_px, y_bottom_px, image_width, image_height):
    # Calculate normalized coordinates
    # This time, the pixels are not used
    x_center_norm = (x_left_px + x_right_px) / (2 * image_width)
    y_center_norm = (y_top_px + y_bottom_px) / (2 * image_height)
    width_norm = (x_right_px - x_left_px) / image_width
    height_norm = (y_bottom_px - y_top_px) / image_height

    # Format the annotation string
    annotation = f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"
    return annotation

def find_annotation(annotation, distance=distance):
    return annotation // distance

def adjust_annotation(y_center, image_part, distance=distance):
    return y_center + image_part * (distance + 1)

def image_split_save(img, y_split, x_left, x_right, original_img, image_part, t):
    img = img[:, x_left:x_right + 1]

    for i in range(len(y_split)-1):
        crop_img = img[y_split[i]:y_split[i + 1], ]

        if i == image_part:
            imageio.imwrite(f"{image_dir}train\\image_split\\{original_img}.png", crop_img)
            # print(f"{image_dir}train\\image_split\\{original_img}.png")
            t=t+1

    return t

def image_rename(img_path):
    match = re.search(r'file_(.*?)\.png', str(img_path))
    if match:
        desired_string = match.group(1)
        return desired_string

def main(image_dir):
    image_dir_path = Path(image_dir)
    n = 0
    t = 0
    for img_path in load_image_files(image_dir_path / "train" / "images"):
        img_path_str = str(img_path)
        n = n+1
        annotation_location = get_annotation_location(img_path)

        image = imageio.imread(img_path)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y_split, x_left, x_right = get_split_point(image, distance)

        image_width = image.shape[1]
        image_height = image.shape[0]
        annotation = parse_and_decode_annotation(annotation_location)
        class_id, x_center_px, y_center_px, width_px, height_px = convert_annotation_to_pixels(annotation, image_width, image_height)
        y_center = return_y_center(annotation, image_height)

        image_part = find_annotation(y_center, distance)
        t = image_split_save(image, y_split, x_left, x_right, image_rename(img_path), image_part, t)

        new_annotation_location = os.path.join(img_path.parent.parent, "labels_adjusted")

        update_annotation(img_path, new_annotation_location, annotation, image_height, y_center)
        print(t)
image_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\"
main(image_dir)