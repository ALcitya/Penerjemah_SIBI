import cv2
import os
import numpy as np

INPUT_DIR = "../data/processed/rgb"
OUTPUT_DIR = "../data/augmented/rgb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):

    label_path = os.path.join(INPUT_DIR, label)
    output_label_path = os.path.join(OUTPUT_DIR, label)

    os.makedirs(output_label_path, exist_ok=True)

    print("Processing label:", label)

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        base_name = os.path.splitext(img_name)[0]

        # ORIGINAL
        cv2.imwrite(os.path.join(output_label_path, base_name + "_orig.jpg"), img)

        # FLIP
        flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_flip.jpg"), flip)

        # ROTATE
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1)
        rotate = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(os.path.join(output_label_path, base_name + "_rot.jpg"), rotate)

        # BRIGHTNESS
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_bright.jpg"), bright)

        # GAUSSIAN NOISE
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_noise.jpg"), noisy)

        # ZOOM
        zoom_factor = 1.2
        zoom = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = zoom.shape[:2]
        zoom_crop = zoom[(zh-h)//2:(zh-h)//2+h, (zw-w)//2:(zw-w)//2+w]
        cv2.imwrite(os.path.join(output_label_path, base_name + "_zoom.jpg"), zoom_crop)

        # GAUSSIAN BLUR
        blur = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_blur.jpg"), blur)

print("Augmentasi RGB selesai!")