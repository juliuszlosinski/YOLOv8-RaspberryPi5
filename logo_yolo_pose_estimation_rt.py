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

# 4. Loading the logo.
logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)  # Load logo with transparency if available.

# Resize logo to fit it in the top-right corner (adjust size as necessary)
logo_height, logo_width = 100, 100  # Adjust logo size here
logo = cv2.resize(logo, (logo_width, logo_height))

# 5. Getting and analyzing output from the camera.
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.3, save=True)
    img = results[0].plot()

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

    cv2.imshow("Camera Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Releasing the handler to the camera.
cap.release()

# 7. Exiting all windows.
cv2.destroyAllWindows()