import cv2
import numpy as np
from src.hand_detection import crop_hand_from_frame

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Tidak dapat membuka file video: {}".format(video_path))
    
    frames = []
    total_frame = 0
    detected_hands = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frame += 1 
        # crop hand
        hand = crop_hand_from_frame(frame)
        if hand is not None:
            frames.append(hand)
            detected_hands += 1

    cap.release()

    print(f"Total frames: {total_frame}, Detected hands: {detected_hands}")
    if len(frames) == 0:
        raise ValueError("Tidak ada tangan yang terdeteksi")
    return np.array(frames)