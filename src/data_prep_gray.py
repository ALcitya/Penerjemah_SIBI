import cv2
import os
import numpy as np
# konfigurasi path
INPUT_DIR = '../data/processed/cleaned_grayscale'
OUTPUT_X = '../data/processed/X_gray.npy'
OUTPUT_Y = '../data/processed/Y_gray.npy'
NUM_FRAMES= 20

def prep_gray():
    X,Y = [],[]
    labels = sorted(os.listdir(INPUT_DIR))
    label_map = {label: i for i, label in enumerate(labels)}
    
    for label in labels:
        label_path = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(label_path):
            continue
        
        print(f"processing grayscale label: {label}")
        all_images = sorted(os.listdir(label_path))
        
        for i in range(0, len(all_images), NUM_FRAMES):
            video_frames = []
            frame_batch = all_images[i : i + NUM_FRAMES]
            if len(frame_batch) < NUM_FRAMES:
                continue
            
            for img_name in frame_batch:
                img = cv2.imread(os.path.join(label_path, img_name), cv2.IMREAD_GRAYSCALE)
                video_frames.append(img.astype('float32') / 255.0) # normalisasi
                
            X.append(video_frames)
            Y.append(label_map[label])
    
    X = np.array(X)
    # tambah dimesnsi channel : (N, 20, 128, 128) -> (N, 20, 128, 128, 1)
    X = np.expand_dims(X, axis= -1)
    
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, np.array(Y))
    print(f"seleai! shape grayescale X: {X.shape}, Y: {len(Y)}")

if __name__ == "__main__":
    prep_gray()