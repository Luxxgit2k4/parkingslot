
import cv2
import torch
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('/home/lakshmanan/parkingspace/yolov5', 'custom',
                       path='/home/lakshmanan/parkingspace/model/best.pt',
                       source='local')

# Function to draw bounding boxes
def draw_boxes(frame, results, confidence_threshold=0.3):
    # Filter the detections to exclude non-parking space and non-car detections
    for result in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        if confidence < confidence_threshold:
            continue  # Ignore detections below confidence threshold

        label = f"{model.names[int(cls)]} {confidence:.2f}"

        # Only detect class for parking spaces (assuming class 0 is for parking spaces)
        if int(cls) == 0:  # Parking space class (free space)
            color = (0, 0, 255)  # Red for free spaces
        elif int(cls) == 1:  # Car class (occupied space)
            color = (0, 255, 0)  # Green for cars (occupied spaces)
        else:
            continue  # Ignore other classes (e.g., face, background)

        # Draw rectangle and label only for parking spaces or cars
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# OpenCV Video Capture (0 for webcam, or replace with video file path)
cap = cv2.VideoCapture(0)  # Change to 'rtsp://your_ip_camera_url' for IP camera feed

if not cap.isOpened():
    print("Error: Unable to access the camera or video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame from feed.")
        break

    # Preprocess the frame for YOLO model
    results = model(frame)

    # Draw bounding boxes and filter detections
    frame_with_boxes = draw_boxes(frame, results)

    # Display the frame with detections
    cv2.imshow("Parking Space Detection", frame_with_boxes)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

