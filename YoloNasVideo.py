import cv2
import torch
import yolo_nas_l  # Import the yolo_nas_l module

from super_gradients.training import models
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Import the YouTubeVideo class from IPython.display
from IPython.display import YouTubeVideo

# Define the YouTube video ID
video_id = 'aE8I7bDf62M'  # Replace YOUR_VIDEO_ID with the actual video ID

# Create a YouTubeVideo object with the specified video ID
video = YouTubeVideo(video_id)

# Display the video
# display(video)

# Define the URL of the YouTube video
video_url = f'https://www.youtube.com/watch?v={video_id}'

# Print a success message
print('Video downloaded successfully')

input_video_path = f"/content/EXTREME SPORTS X DIVERSE-{video_id}.mp4"
output_video_path = "detections.mp4"

device = 'cuda' if torch.cuda.is_available() else "cpu"

yolo_nas_l.to(device).predict(input_video_path).save(output_video_path)