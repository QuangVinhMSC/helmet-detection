import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
from sort import *

# Create an instance of SORT
mot_tracker = Sort()

# Load a pre-trained YOLO model
model = YOLO("./model_weight/yolo-v8n-version1.pt")
logger.info("Init model success. Ready for predicting")

# Open the input video
input_video = cv2.VideoCapture("./sample_image/video_demo/IMG_1099.mp4")

# Get video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(input_video.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output_video = cv2.VideoWriter(
    './sample_image/output/IMG_1099.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Process each frame
while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    detections = model.predict(source=frame, show=False)

    # Initialize an empty list to store the formatted detections
    formatted_detections = []

    # Loop through each detected object
    for detection in detections[0].boxes:
        # Extract the bounding box coordinates and confidence score
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        score = detection.conf[0].tolist()
        formatted_detections.append([x1, y1, x2, y2, score])

    formatted_detections_array = np.array(formatted_detections)

    # Update SORT tracker with the detections
    track_bbs_ids = mot_tracker.update(formatted_detections_array)

    # Loop through each tracked object and draw the rectangle with the ID
    for track in track_bbs_ids:
        x1, y1, x2, y2, track_id = track.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_video.write(frame)

input_video.release()
output_video.release()
cv2.destroyAllWindows()

logger.info("Video processing completed and saved as output_video.mp4")
