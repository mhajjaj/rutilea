import cv2

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

# Load the image
image = cv2.imread("GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\images\\file_year_2022_month_07_day_01_19_01_06_NG_2_8.png")

# Path to the annotation file associated with an image
annotation_file_path = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\labels\\file_year_2022_month_07_day_01_19_01_06_NG_2_8.txt"

# Read and process annotation lines from the file
with open(annotation_file_path, "r") as file:
    for annotation_line in file:
        # Call the parse_and_decode_annotation function
        class_id, x_center, y_center, width, height = parse_and_decode_annotation(annotation_line.strip())

        # Now you can use these values to draw bounding boxes or perform other tasks.
        # For example, you can draw bounding boxes on the image or display class labels.
        # Make sure you have the corresponding image loaded as well.
        print(f"Class ID: {class_id}, x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height}")

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

# Draw bounding box and overlay label
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle
label_text = f"Class: {class_id}"
cv2.putText(image, label_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

# Display the image
# Save the annotated image to a file
cv2.imwrite("annotated_image.jpg", image)