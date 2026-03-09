import cv2
import os

root_raw_dir = '../data/raw'
output_root = '../data/processed'

num_frames_to_save = 20
target_size = (128,128)

os.makedirs(os.path.join(output_root, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'grayscale'), exist_ok=True)

for label in os.listdir(root_raw_dir):

    word_path = os.path.join(root_raw_dir, label)

    if not os.path.isdir(word_path):
        continue

    print(f"Processing label: {label}")

    rgb_label_dir = os.path.join(output_root, 'rgb', label)
    gray_label_dir = os.path.join(output_root, 'grayscale', label)

    os.makedirs(rgb_label_dir, exist_ok=True)
    os.makedirs(gray_label_dir, exist_ok=True)

    for video_name in os.listdir(word_path):

        if not video_name.lower().endswith('.webm'):
            continue

        video_full_path = os.path.join(word_path, video_name)
        video_base_name = os.path.splitext(video_name)[0]

        cap = cv2.VideoCapture(video_full_path)

        if not cap.isOpened():
            print(f"Gagal membuka video: {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Video rusak: {video_name}")
            cap.release()
            continue

        step = max(1, total_frames // num_frames_to_save)

        for i in range(num_frames_to_save):

            frame_id = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = cap.read()
            if not ret:
                break

            frame_res = cv2.resize(frame, target_size)
            # crop bagian atas
            h, w = frame_res.shape[:2]
            crop_top = int(h * 0.25)
            frame_crop = frame_res[crop_top:h, 0:w]

            # resize lagi agar tetap 128x128
            frame_crop = cv2.resize(frame_crop, target_size)

            rgb_name = f"{video_base_name}_f{i:02d}.jpg"
            cv2.imwrite(os.path.join(rgb_label_dir, rgb_name), frame_crop)

            gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            gray_name = f"{video_base_name}_f{i:02d}.jpg"
            cv2.imwrite(os.path.join(gray_label_dir, gray_name), gray)

        cap.release()

        print(f"   selesai: {video_name}")

print("\nSemua video selesai diproses.")