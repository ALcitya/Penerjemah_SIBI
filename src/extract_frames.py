import cv2
from hand_detection import crop_hand_from_frame

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hand = crop_hand_from_frame(frame)

        if hand is not None:
            frames.append(hand)

    cap.release()

    print("Total frame tangan:", len(frames))

    return frames