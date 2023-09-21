import cv2
import os

# Directory paths for annotated images and output directory for cleaned images
annotated_image_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\labels_annotated"
output_dir = "GearInspection-Dataset3\\CategoryNG\\ClassAll\\train\\images_cleaned"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of annotated image files in the annotated image directory
annotated_image_files = [f for f in os.listdir(annotated_image_dir) if f.endswith("_annotated.png")]

# Process each annotated image
for annotated_image_file in annotated_image_files:
    # Load the annotated image
    annotated_image_path = os.path.join(annotated_image_dir, annotated_image_file)
    annotated_image = cv2.imread(annotated_image_path)

    # Remove any annotation drawings or overlays (e.g., bounding boxes and labels) from the image
    # You can add code here to remove specific annotations based on your annotations' format

    # Save the cleaned image to the output directory with the original filename
    cleaned_image_file = annotated_image_file.replace("_annotated.png", ".png")
    cleaned_image_path = os.path.join(output_dir, cleaned_image_file)
    cv2.imwrite(cleaned_image_path, annotated_image)
