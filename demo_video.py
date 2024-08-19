import cv2
import tempfile
from ultralytics import YOLO
from loguru import logger
import cv2


# Load a pre-trained YOLO model
model = YOLO("./model_weight/yolo-v8n-version1.pt")
logger.info("Init model success. Ready for predicting")

# Path to the video file
video_path = './home/trieu/Documents/SIC/sample_image/output'
output_video_path = './sample_image/output/clip.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def process_frame(image, results):
    '''
    With results from yolo model, draw the bounding box of objects in that frame
    '''
    color4cls0 = (255,0,0)
    color4cls1 = (0,255,0)
    color4cls2 = (0,0,255)
    color4cls3 = (125,125,125)
    
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    names = results[0].names
    confidences = results[0].boxes.conf

    for ids, box in enumerate(boxes):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))

        if int(classes[ids]) == 0:
            name = names[0]
            color = color4cls0
        elif int(classes[ids]) == 1:
            name = names[1]
            color = color4cls1
        elif int(classes[ids]) == 2:
            name = names[2]
            color = color4cls2
        elif int(classes[ids]) == 3:
            name = names[3]
            color = color4cls3

        cv2.rectangle(image, start_point, end_point, color=color, thickness=2)
        cv2.putText(
            image,
            str(name),
            (int(box[0]), int(box[1]) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=3
        )
    return image

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

logger.info("Prediction is on going")
while True:
    frames = frames + 1
    ret, frame = cap.read()

    # If the frame was not successfully read, it means we've reached the end of the video
    if not ret:
        print("Reached the end of the video.")
        break

    # results = model.predict(source = frame,verbose=False) 
    # frame = process_frame(frame,results)

    # Write the processed frame to the output video
    out.write(frame)

    # cv2.imshow('Processed Frame', processed_frame)

    # Wait for 25 ms and check if the user pressed 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
logger.info(f"Output is saved in {output_video_path}")


