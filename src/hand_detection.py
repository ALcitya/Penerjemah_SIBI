import cv2
from ultralytics import YOLO

# load model sekali
try:
    yolo_model = YOLO('yolov8n-hand.pt')
except:
    yolo_model = YOLO('yolov8n.pt')

def crop_hand_from_frame(frame, target_size=(128,128)):

    results = yolo_model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            hand = frame[y1:y2, x1:x2]

            if hand.shape[0] > 0 and hand.shape[1] > 0:
                hand = cv2.resize(hand, target_size)
                return hand / 255.0

    return None