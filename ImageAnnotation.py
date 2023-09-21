import cv2
import os

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

# Directory paths for images and labels
image_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\images"
label_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\labels"
output_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\images_labels_split_annotated"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

# Process each image-label pair
for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    # Construct the corresponding annotation file path
    annotation_file = os.path.splitext(image_file)[0] + ".txt"
    annotation_file_path = os.path.join(label_dir, annotation_file)

    # Read and process annotation lines from the file
    with open(annotation_file_path, "r") as file:
        for annotation_line in file:
            # Call the parse_and_decode_annotation function
            class_id, x_center, y_center, width, height = parse_and_decode_annotation(annotation_line.strip())

            # Now you can use these values to draw bounding boxes or perform other tasks.
            # For example, you can draw bounding boxes on the image or display class labels.
            # Make sure you have the corresponding image loaded as well.
            print(f"Image: {image_file}, Class ID: {class_id}, x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height}")

    # Parse and decode the annotation (assuming annotation in YOLO format)
    class_id, x_center, y_center, width, height = parse_and_decode_annotation(annotation_line)

    # Convert normalized coordinates to pixel values
    image_height, image_width, _ = image.shape
    x_center_px = int(x_center * image_width)
    y_center_px = int(y_center * image_height)
    width_px = int(width * image_width)
    height_px = int(height * image_height)

    # Calculate bounding box coordinates
    top_left = (x_center_px - width_px // 2, y_center_px - height_px // 2)
    bottom_right = (x_center_px + width_px // 2, y_center_px + height_px // 2)

    print(image.shape)
    print(x_center, y_center)
    print(x_center_px, y_center_px)
    exit()
    # Draw bounding box and overlay label
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle
    label_text = f"Class: {class_id}"
    cv2.putText(image, label_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

    # Save the annotated image with a modified filename
    annotated_image_file = os.path.splitext(image_file)[0] + "_annotated.png"
    annotated_image_path = os.path.join(output_dir, annotated_image_file)
    cv2.imwrite(annotated_image_path, image)