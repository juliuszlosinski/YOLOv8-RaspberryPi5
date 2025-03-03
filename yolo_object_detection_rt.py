# 0. Importing libraries.
from ultralytics import YOLO
import cv2
import math
import time
import random

# 1. Setting up class names.
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# 2. Generating random color for each class.
def generate_color(seed):
    random.seed(seed)  # Fix the seed for consistent results
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
class_colors = {i: generate_color(i) for i in range(len(class_names))}

# 3. Loading the YOLOv8 nano model for object detection.
model = YOLO("yolov8n.pt")

# 4. Getting handler to the camera by using OpenCV library.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 5. Setting up parameters of the windows.
cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 6. Getting and analyzing output from the camera.
while True:
    success, img = cap.read()
    if not success:
        break  
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])  
            color = class_colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            print("Confidence: ", confidence)
            print("Class name: ", class_Names[cls])
            cv2.putText(img, class_names[cls], org, font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# 7. Releasing the handler to the camera.
cap.release()

# 8. Exiting all windows.
cv2.destroyAllWindows()