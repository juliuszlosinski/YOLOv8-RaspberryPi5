# 0. Importing libraries.
from ultralytics import YOLO
import cv2

# 1. Loading YOLOv8 pose estimation model.
model = YOLO("yolov8n-pose.pt")

# 2. Getting handler to the camera by using OpenCV library.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 3. Setting parameters of the window.
cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 4. Getting and analyzing output from the camera.
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.3, save=True)
    img = results[0].plot() 
    cv2.imshow("Camera Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Releasing the handler to the camera.
cap.release()

# 6. Exiting all windows.
cv2.destroyAllWindows()