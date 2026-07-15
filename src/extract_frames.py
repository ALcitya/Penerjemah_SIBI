import cv2
import numpy as np
from src.hand_detection import crop_hand_from_frame

def extract_frames(VIDEO_PATH, max_frames=100):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError("Tidak dapat membuka file video: {}".format(VIDEO_PATH))
    
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
            # berhenti jika jumlah frame tercapai
            if len(frames) >= max_frames:
                break

    cap.release()

    print(f"Total frames: {total_frame}, Detected hands: {detected_hands}")
    if len(frames) == 0:
        raise ValueError("Tidak ada tangan yang terdeteksi")
    return np.array(frames)