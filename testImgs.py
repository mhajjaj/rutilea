import os
import cv2

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
        print(f"Dimensions: {width}x{height}")
        print(f"Number of Channels: {channels}")

        # You can also access the pixel data of the image using img
        # For example, to access the pixel at (x, y):
        # pixel = img[y, x]

        # If you want to perform more specific operations on the image content, you can do so here.

        # Close the image (not necessary, but it's a good practice)
        cv2.destroyAllWindows()
    else:
        print(f"Unable to read file: {png_file}")
