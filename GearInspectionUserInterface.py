# from super_gradients.training import models
# import os

# HOME = os.getcwd()
# MODEL_PATH = f'{HOME}\\checkpoint\\AGI-Dataset\\AGIExperiment\\average_model.pth'
# NUM_CLASSES = 8

# img_path = f'{HOME}\\GearInspection-Dataset3\\CategoryNG\\Results\\ForPredict\\OK\\year=2023-month=04-day=01-00_54_56-OK-2_5.png'
# best_model = models.get('yolo_nas_l', num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)
# best_model.predict(img_path, conf=0.25).show()

import os
import random
from super_gradients.training import models

HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, 'checkpoint', 'AGI-Dataset', 'AGIExperiment', 'average_model.pth')
NUM_CLASSES = 8
CONF = 0.25

# Path to the folder containing the images
image_folder = os.path.join(HOME, 'GearInspection-Dataset3', 'CategoryNG', 'ForPredict', 'OK')

# Verify if the folder exists
if not os.path.exists(image_folder):
    print(f"The specified folder '{image_folder}' does not exist.")
else:
    # Specify the folder path to save the prediction results
    results_folder = os.path.join(HOME, 'GearInspection-Dataset3', 'PredictionResults')

    # Create the results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    # List all image files in the folder
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.png')]

    # Randomly select 10 images from the list
    selected_images = random.sample(image_files, min(10, len(image_files)))

    # Initialize the model
    best_model = models.get('yolo_nas_l', num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)

    # Iterate through the selected images and make predictions
    for img_path in selected_images:
        prediction = best_model.predict(img_path, conf=CONF)
        
        # Save the predicted image directly to the results folder
        result_filename = os.path.basename(img_path)
        result_path = os.path.join(results_folder, result_filename)
        prediction.save(result_filename)

        # Show the prediction
        prediction.show()

    print(f"Predicted images saved in the '{results_folder}' folder.")