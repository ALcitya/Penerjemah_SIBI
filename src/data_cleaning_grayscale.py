import cv2
import os
import numpy as np

# konfigurasi path
input_base_dir = '../data/processed/grayscale'
output_base_dir = '../data/processed/cleaned_grayscale'

def apply_clahe(image):
    # meningkatkan kontras gambar
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def is_too_dark(image, threshold=20):
    return np.mean(image) < threshold

# iterasi setiap label
for label in os.listdir(input_base_dir):
    label_path = os.path.join(input_base_dir, label)
    
    if os.path.isdir(label_path):
        out_label_path = os.path.join(output_base_dir, label)
        os.makedirs(out_label_path, exist_ok=True)
        
        print(f"Cleaning grayscale images for label: {label}")
        
        # 2. iterasi setiap frame dalam label
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            # cleaning
            # 1. buang jika terlalu gelap
            if is_too_dark(img):
                print(f"Removed dark image: {img_name}")
                continue
            # 2. terapkan CLAHE untuk memperjelas gerakan tangan
            cleaned_img = apply_clahe(img)
            # simpan hasil gambar
            cv2.imwrite(os.path.join(out_label_path, img_name), cleaned_img)
print("cleaning gambar grayscale selesai.")