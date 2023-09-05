import os
import cv2
import torch
from super_gradients.training import models

# Directory containing the PNG files
folder_path = 'predictImgs\\Scratches\\'

# List all files in the folder
file_list = os.listdir(folder_path)

# Filter the list to include only PNG files
png_files = [file for file in file_list if file.lower().endswith('.png')]

# Loop through the PNG files and read their content
for png_file in png_files:
    file_path = os.path.join(folder_path, png_file)

    # Read the PNG file using OpenCV
    img = cv2.imread(file_path)

    if img is not None:
        # Get image dimensions
        height, width, channels = img.shape

        # Print some information about the image
        print(f"File: {png_file}")
        #print(f"Dimensions: {width}x{height}")
        #print(f"Number of Channels: {channels}")

        # Check if the file exists (optional)
        if os.path.exists(file_path):
            # File exists; proceed with the operations.
            print(f"Processing file: {file_path}")
        
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
        
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = model.predict(img)
            out.show()
            print("Done")
        else:
            print(f"File not found: {file_path}")

        cv2.destroyAllWindows()
    else:
        print(f"Unable to read file: {png_file}")