import cv2
import numpy as np
from src.hand_detection import crop_hand_from_frame

def extract_frames(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Tidak dapat membuka file video: {}".format(video_path))

    crops = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        crop = crop_hand_from_frame(frame)   # crop LANGSUNG, frame asli tidak disimpan
        crops.append(crop)
        del frame   # bebaskan frame resolusi penuh segera

        if len(crops) >= max_frames:
            break

    cap.release()

    if len(crops) == 0:
        raise ValueError("Tidak ada frame yang dapat dibaca")

    detected_hands = sum(1 for c in crops if c is not None)
    print(f"Total Frames: {total_frames}, Detected Hands(real): {detected_hands}/{len(crops)} "
          f"After Filling: {len(crops)}")
    if detected_hands == 0:
        raise ValueError("Tidak ada tangan yang terdeteksi dalam video")

    # isi frame yang tidak terdeteksi (forward + backward fill)
    last_good = None
    for i in range(len(crops)):
        if crops[i] is not None:
            last_good = crops[i]
        elif last_good is not None:
            crops[i] = last_good

    next_good = None
    for i in range(len(crops) - 1, -1, -1):
        if crops[i] is not None:
            next_good = crops[i]
        elif next_good is not None:
            crops[i] = next_good
    
    print(f"After Filling: {len(crops)} frame")
    return np.array(crops)