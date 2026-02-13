import cv2
import os

# Konfigurasi Path
root_raw_dir = '../data/raw'
output_root = '../data/processed'
num_frames_to_save = 20
target_size =(128,128)

# folder tujuan
os.makedirs(os.path.join(output_root, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'grayscale'), exist_ok=True)

# looping
# masuk kedalam folder raw
for label in os.listdir(root_raw_dir):
    word_path = os.path.join(root_raw_dir, label)
    
    # pastikan itu folder
    if os.path.isdir(word_path):
        print(f"processing label: {label}")
        
        # membuat subfolder di folder output
        rgb_label_dir = os.path.join(output_root, 'rgb', label)
        gray_label_dir = os.path.join(output_root, 'grayscale', label)
        os.makedirs(rgb_label_dir, exist_ok=True)
        os.makedirs(gray_label_dir, exist_ok=True)
        
        # loop setiap video .webm
        for video_name in os.listdir(word_path):
            if video_name.endswith('.webm'):
                video_full_path = os.path.join(word_path, video_name)
                video_base_name = os.path.splitext(video_name)[0]
                
                cap = cv2.VideoCapture(video_full_path, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print(f"Gagal membuka video: {video_name}")
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Hindari pembagian dengan nol jika video rusak
                step = max(1, total_frames // num_frames_to_save)
                
                # ekstrak 20 frame
                for i in range(num_frames_to_save):
                    frame_id = i * step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Pemrosesan: Resize
                    frame_res = cv2.resize(frame, target_size)
                    
                    # Format RGB (Simpan sebagai BGR untuk cv2.imwrite)
                    rgb_filename = f"{video_base_name}_f{i:02d}.jpg"
                    cv2.imwrite(os.path.join(rgb_label_dir, rgb_filename), frame_res)
                    
                    # Format Grayscale
                    frame_gray = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY)
                    gray_filename = f"{video_base_name}_f{i:02d}.jpg"
                    cv2.imwrite(os.path.join(gray_label_dir, gray_filename), frame_gray)
                
                cap.release()
                print(f"   - Selesai: {video_name}")

print("\n[PROSES SELESAI] Semua video telah diekstrak ke folder processed.")
                            
                    
                