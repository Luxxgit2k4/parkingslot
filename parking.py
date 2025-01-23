
import cv2
import torch
import numpy as np

model = torch.hub.load('/home/lakshmanan/parkingspace/yolov5', 'custom',
                       path='/home/lakshmanan/parkingspace/model/best.pt',
                       source='local')

def process_detections(frame, results, confidence_threshold=0.3):
    total_spaces = 0
    filled_spaces = 0
    data = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        if confidence < confidence_threshold:
            continue

        cls = int(cls)
        if cls == 0:
            total_spaces += 1
            data.append(0)
            color = (0, 0, 255)
        elif cls == 1:
            total_spaces += 1
            filled_spaces += 1
            data.append(1)
            color = (0, 255, 0)
        else:
            continue

        label = f"{model.names[cls]} {confidence:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    return frame, output

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera or video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame from feed.")
        break

    results = model(frame)
    frame_with_boxes, parking_data = process_detections(frame, results)
    print(parking_data)
    cv2.imshow("Parking Space Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

