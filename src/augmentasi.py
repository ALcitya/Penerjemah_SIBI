import cv2
import os
import numpy as np

INPUT_DIR = "../data/hands"
OUTPUT_DIR = "../data/augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):

    label_path = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(label_path):
        continue

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

        # BRIGHTNESS (Adjust Alpha for brightness and Beta for contrast)
        brightness_factor = 1.2 # Increase brightness by 20%
        bright_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_bright.jpg"), bright_img)

        # CONTRAST
        contrast_factor = 1.5 # Increase contrast by 50%
        contrast_img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
        cv2.imwrite(os.path.join(output_label_path, base_name + "_contrast.jpg"), contrast_img)

        # TRANSLATION (Shift horizontally)
        shift_x = 10 # Shift 10 pixels to the right
        M_trans = np.float32([[1, 0, shift_x], [0, 1, 0]])
        translated_img = cv2.warpAffine(img, M_trans, (w, h))
        cv2.imwrite(os.path.join(output_label_path, base_name + "_trans.jpg"), translated_img)

        # 5 New Background Color Augmentations (tinting)
        # Define 5 background colors (BGR format) for blending
        bg_colors = [
            (0, 0, 200),  # Reddish tint
            (0, 200, 0),  # Greenish tint
            (200, 0, 0)  # Bluish tint
        ]
        color_names = ["redtint", "greentint", "bluetint"]
        alpha = 0.7 # Weight of the original image (0.7 means 70% original, 30% new color)

        for i, (b_color, name) in enumerate(zip(bg_colors, color_names)):
            # Create a solid color background image of the same size as img
            solid_bg = np.full(img.shape, b_color, dtype=np.uint8)
            # Blend the original image with the solid color background
            blended_img = cv2.addWeighted(img, alpha, solid_bg, 1 - alpha, 0)
            cv2.imwrite(os.path.join(output_label_path, f"{base_name}_{name}.jpg"), blended_img)

print("Augmentasi RGB selesai!")