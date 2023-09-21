import numpy as np
import imageio.v2 as imageio
import cv2
from scipy.signal import find_peaks
from pathlib import Path
import re

image_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\"

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

    # print(y_split, _ )
    # exit()

    edges = cv2.Canny(np.array(img), 50, 80)
    profile_x = np.mean(edges, axis=0)
    x_right = np.where(profile_x > 5)[0][0]
    x_left = np.where(profile_x[0:img.shape[1] - 100] > 10)[0][-1]

    return y_split, x_right, x_left

def parse_and_decode_annotation(annotation_line):
    # Split the annotation line into individual values
    values = annotation_line.split()

    # Extract class ID and convert it to an integer
    class_id = int(values[0])

    # Extract other values and convert them to floats
    x_center = float(values[1])
    y_center = float(values[2])
    width = float(values[3])
    height = float(values[4])

    return class_id, x_center, y_center, width, height

def get_annotation_location(original_annotation, split_coordinates):
    # Parse the original annotation values
    class_id, x_center, y_center, width, height = parse_and_decode_annotation(original_annotation)

    print(class_id, x_center, y_center, width, height)
    exit()

    # Iterate through split regions and find the one that contains the object's center
    for i in range(len(split_coordinates) - 1):
        if split_coordinates[i] <= y_center < split_coordinates[i + 1]:
            # Calculate the adjusted coordinates within the split region
            new_y_center = y_center - split_coordinates[i]
            
            # Adjust the y-coordinate relative to the split region
            new_annotation = f"{class_id} {x_center} {new_y_center} {width} {height}"

            # Return the adjusted annotation
            return new_annotation

    # If the object's center is not within any split region, return None
    return None

def image_split_save(img, y_split, x_left, x_right, n, original_img):
    img = img[:, x_left:x_right + 1]

    for i in range(len(y_split)-1):
        crop_img = img[y_split[i]:y_split[i + 1], ]

        # Save the split image
        imageio.imwrite(f"{image_dir}train\\image_split\\{original_img}_{n:03}.png", crop_img)
############# find_annotation_on_the_image():
        # Create and save the corresponding annotation for the split image
        # save_annotation(original_img, n, x_left, x_right, y_split[i], y_split[i + 1])
        n += 1

    print(n, x_left, x_right, y_split[i], y_split[i + 1])
    exit()

    return n

def save_annotation(original_img_path, n, x_left, x_right, y_top, y_bottom):
    # Load the original image
    # print(f"file_{original_img_path}.png")
    #exit()
    original_img = imageio.imread(f"{image_dir}train\\images\\file_{original_img_path}.png")

    # Create a Path object from the original image path
    original_img_path = Path(original_img_path)
    
    # Extract the stem (base name without extension) of the file
    file_stem = original_img_path.stem
    
    # Calculate annotation values based on the split coordinates
    width = (x_right - x_left) / original_img.shape[1]
    height = (y_bottom - y_top) / original_img.shape[0]
    x_center = (x_left + x_right) / (2 * original_img.shape[1])
    y_center = (y_top + y_bottom) / (2 * original_img.shape[0])
    
    # Create and save the annotation file
    annotation_file_path = f"{image_dir}train\\labels_annotated\\{file_stem}_{n:03}.txt"
    with open(annotation_file_path, "w") as file:
        file.write(f"0 {x_center} {y_center} {width} {height}")

def image_rename(img_path):
    match = re.search(r'file_(.*?)\.png', str(img_path))
    if match:
        desired_string = match.group(1)
        # print(desired_string)
        return desired_string

def main(input_directory):
    n = 0
    input_directory_path = Path(input_directory)
    for img_path in load_image_files(input_directory_path):
        img = imageio.imread(img_path)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y_split, x_left, x_right = get_split_point(img)

        if not (683 < x_right - x_left < 695):
            print(">>> ", x_right - x_left)
            continue

        for i in range(len(y_split)-1):
            if not (308 < y_split[i + 1] - y_split[i] < 320):
                print(">>> ", y_split[i + 1] - y_split[i])
                continue

        n = image_split_save(img, y_split, x_left, x_right, n, image_rename(img_path))

main(f"{image_dir}train\\images")

# Example usage:
original_annotation = "0 0.5 0.7 0.2 0.3"  # Example annotation from XX.txt
split_coordinates = [0.2, 0.5, 0.8]  # Example split points from get_split_point

adjusted_annotation = get_annotation_location(original_annotation, split_coordinates)
if adjusted_annotation is not None:
    print("Adjusted Annotation:", adjusted_annotation)
else:
    print("Annotation not within any split region.")

