import cv2
import os
import numpy as np

# konfigurasi path
input_rgb_dir = '../data/processed/rgb'
output_rgb_dir = '../data/processed/cleaned_rgb'

def is_blurry(image, threshold=100):
    # mendeteksi blur menggunakan variance of Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def clean_rgb_data(input_dir, output_dir):
    for label in os.listdir(input_dir):
        path = os.path.join(input_dir, label)
        if os.path.isdir(path):
            save_path = os.path.join(output_dir, label)
            os.makedirs(save_path, exist_ok=True)
            
            for img_name in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_name))
                if img is None:
                    continue
                
                # cleaning rgb
                # 1. buang jika blur
                if is_blurry(img, threshold=50):
                    continue
                # 2. buang jika terlalu gelap
                avg_brightness = np.mean(img)
                if avg_brightness < 30 or avg_brightness > 240:
                    continue
                # 3. penajaman gambar
                kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                sharpened = cv2.filter2D(img, -1, kernel)
                
                cv2.imwrite(os.path.join(save_path, img_name), sharpened)
                
clean_rgb_data(input_rgb_dir, output_rgb_dir)