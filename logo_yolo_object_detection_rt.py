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

# 5. Loading the logo.
logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)  # Load logo with transparency if available.

# Resize logo to fit it in the top-right corner (adjust size as necessary)
logo_height, logo_width = 100, 100  # Adjust logo size here
logo = cv2.resize(logo, (logo_width, logo_height))

# 6. Setting up parameters of the windows.
cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 7. Getting and analyzing output from the camera.
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
            print("Class name: ", class_names[cls])
            cv2.putText(img, class_names[cls], org, font, fontScale, color, thickness)
            
    top_left_x = img.shape[1] - logo_width - 10 
    top_left_y = 10  
    
    if logo.shape[2] == 4:
        
        alpha_logo = logo[:, :, 3] / 255.0
        logo_rgb = logo[:, :, :3]
        roi = img[top_left_y:top_left_y+logo_height, top_left_x:top_left_x+logo_width]
        
        for c in range(0, 3):
            roi[:, :, c] = (alpha_logo * logo_rgb[:, :, c] + (1 - alpha_logo) * roi[:, :, c])
        img[top_left_y:top_left_y+logo_height, top_left_x:top_left_x+logo_width] = roi
    else:
        img[top_left_y:top_left_y+logo_height, top_left_x:top_left_x+logo_width] = logo
        
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# 8. Releasing the handler to the camera.
cap.release()

# 9. Exiting all windows.
cv2.destroyAllWindows()