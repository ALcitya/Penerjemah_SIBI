import cv2
import os
import numpy as np

INPUT_DIR = '../data/augmented/grayscale'
OUTPUT_X = '../data/prep/grayscale/X_gray.npy'
OUTPUT_Y = '../data/prep/grayscale/Y_gray.npy'

NUM_FRAMES = 20
IMG_SIZE = 128

def prep_gray():

    X, Y = [], []

    labels = sorted(os.listdir(INPUT_DIR))
    label_map = {label: i for i, label in enumerate(labels)}

    for label in labels:

        label_path = os.path.join(INPUT_DIR, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")

        all_images = sorted([
            f for f in os.listdir(label_path)
            if f.endswith(".png") or f.endswith(".jpg")
        ])

        for i in range(0, len(all_images), NUM_FRAMES):

            frame_batch = all_images[i:i+NUM_FRAMES]

            if len(frame_batch) < NUM_FRAMES:
                continue

            video_frames = []

            for img_name in frame_batch:

                path = os.path.join(label_path, img_name)

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                img = img.astype('float32') / 255.0

                video_frames.append(img)

            X.append(video_frames)
            Y.append(label_map[label])

    X = np.array(X)
    Y = np.array(Y)

    X = np.expand_dims(X, axis=-1)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, Y)

    print("Selesai!")
    print("Shape X:", X.shape)
    print("Shape Y:", Y.shape)

if __name__ == "__main__":
    prep_gray()