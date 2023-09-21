import numpy as np
import imageio.v2 as imageio
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path



def load_image_files(directory_path):
    """Load all PNG image files from the specified directory."""
    return list(Path(directory_path).glob("*.png"))


def make_ground_truth(raw_image_path, base_dir):
    "ground truthを作成する"
    base_dir = Path(base_dir)
    text_dir = base_dir / "labels"

    with open(str(text_dir / f"{raw_image_path.stem}.txt"), "r") as f:
        image = imageio.imread(raw_image_path)
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape
        canvas = np.zeros_like(image, dtype=np.uint8)
        for line in f:
            staff = line.split()
            x_center, y_center, w, h = float(
                staff[1]) * width, float(staff[2]) * height, float(staff[3]) * width, float(staff[4]) * height
            x1 = round(x_center - w / 2)
            y1 = round(y_center - h / 2)
            x2 = round(x_center + w / 2)
            y2 = round(y_center + h / 2)

            canvas[y1:y2, x1:x2] = 255

    return canvas


def get_split_point(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded_img = cv2.erode(img, kernel, iterations=1)
    profile_y = np.mean(eroded_img, axis=1)
    y_split, _ = find_peaks(-profile_y, distance=300)

    edges = cv2.Canny(np.array(img), 50, 80)
    profile_x = np.mean(edges, axis=0)
    x_right = np.where(profile_x > 5)[0][0]
    x_left = np.where(profile_x[0:img.shape[1] - 100] > 10)[0][-1]

    return y_split, x_right, x_left


def image_split_save(img, mask_img, y_split, x_left, x_right, defect, n):
    img = img[:, x_left:x_right + 1]
    mask_img = mask_img[:, x_left:x_right + 1]

    for i in range(len(y_split)-1):
        crop_img = img[y_split[i]:y_split[i + 1], ]
        crop_mask_img = mask_img[y_split[i]:y_split[i + 1], ]


        if np.mean(crop_mask_img) > 0:

            imageio.imwrite(f"dataset_for_patchcore/test/{defect}/{n:03}.png", crop_img)
            imageio.imwrite(f"dataset_for_patchcore/groundtruth/{defect}/{n:03}_mask.png", crop_mask_img)

            n += 1

    return n


def main(input_directory):
    n = 0
    input_directory_path = Path(input_directory)
    for img_path in load_image_files(input_directory_path / "images"):
        mask_img = make_ground_truth(img_path, input_directory)

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

        last_element = input_directory.split('_')[-1]
        if 'kizu' in last_element:
            defect = 'scratch'
        elif 'dakon' in last_element:
            defect = 'bruise'
        elif 'akkon' in last_element:
            defect = 'indentation'


        n = image_split_save(img, mask_img, y_split, x_left, x_right, defect, n)


directories = glob.glob("yolo/yolo/*")
for directory in directories:
    main(directory)