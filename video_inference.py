import cv2 as cv
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO(r"C:\Users\ahmed\Downloads\best (12).pt")

# Path to input video
input_video_path = "input video path"

# Path to save output video
output_video_path = "output video path"

# Open the video capture
cap = cv.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'mp4v')

# Define the video writer
out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Inference on each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)
    result_frame = results[0].plot()

    # Write the frame to the output video
    out.write(result_frame)

# Release everything
cap.release()
out.release()
cv.destroyAllWindows()
print("Video processing complete. Output saved at:", output_video_path)
