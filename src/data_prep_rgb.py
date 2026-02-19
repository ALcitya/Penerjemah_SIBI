import cv2
import os
import numpy as np
# konfigurasi path
INPUT_DIR = '../data/processed/cleaned_rgb'
OUTPUT_X = '../data/processed/X_rgb.npy'
OUTPUT_Y = '../data/processed/Y_rgb.npy'
NUM_FRAMES= 20
def prepare_rgb():
    X, Y = [], []
    labels = sorted(os.listdir(INPUT_DIR))
    label_map = {label: i for i, label in enumerate(labels)}

    for label in labels:
        label_path = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(label_path): continue
        
        print(f"Processing RGB: {label}")
        all_images = sorted(os.listdir(label_path))
        
        for i in range(0, len(all_images), NUM_FRAMES):
            video_frames = []
            frame_batch = all_images[i : i + NUM_FRAMES]
            if len(frame_batch) < NUM_FRAMES: continue

            for img_name in frame_batch:
                img = cv2.imread(os.path.join(label_path, img_name)) # Default BGR/RGB
                video_frames.append(img.astype('float32') / 255.0) # Normalisasi
            
            X.append(video_frames)
            Y.append(label_map[label])

    X = np.array(X)
    # Shape sudah (N, 20, 128, 128, 3) secara otomatis
    
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, np.array(Y))
    print(f"Selesai! Shape RGB: {X.shape}")

if __name__ == "__main__":
    prepare_rgb()