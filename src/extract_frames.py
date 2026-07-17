import cv2
import numpy as np
from src.hand_detection import crop_hand_from_frame

def extract_frames(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Tidak dapat membuka file video: {}".format(video_path))
    # 1. baca semua frame mentah
    raw_frames = []
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames +=1
        raw_frames.append(frame)
        if len(raw_frames) >= max_frames:
            break
    cap.release()
    
    if len(raw_frames) ==0:
        raise ValueError("Tidak ada frame yang dapat dibaca")
    # 2. Deteksi tangan tiap frame
    crops =[crop_hand_from_frame(f) for f in raw_frames]
    detected_hands = sum(1 for c in crops if c is not None)
    
    if detected_hands ==0:
        raise ValueError("Tidak ada tangan yang terdeteksi dalam video")
    # 3. isi frame yang tidak terdeteksi dengan frame sebelumnya
    last_good= None
    for i in range(len(crops)):
        if crops[i] is not None:
            last_good = crops[i]
        elif last_good is not None:
            crops[i] = last_good
    next_good = None
    for i in range(len(crops) -1,-1,-1):
        if crops[i] is not None:
            next_good = crops[i]
        elif next_good is not None:
            crops[i]= next_good
    print(f"Total Frames: {total_frames}, Detected Hands(real): {detected_hands}/{len(crops)}"
          f"After Filling: {len(crops)}")
    return np.array(crops)
            