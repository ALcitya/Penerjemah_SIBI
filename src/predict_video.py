import numpy as np
import os
from collections import Counter
from tensorflow.keras.models import load_model
from extract_frames import extract_frames

# CONFIG
VIDEO_PATH = "./data/videos/video_gerakan_1.mp4"
MODEL_PATH = "./models/sibi_model_rgb.keras"
DATASET_PATH = "./data/processed/rgb"
SEQ_LEN = 20

# LOAD MODEL & LABEL
model = load_model(MODEL_PATH, compile=False)
labels = sorted(os.listdir(DATASET_PATH))

print("Labels:", labels)

# SLIDING WINDOW
def sliding_window(frames, seq_len=20, step=5):

    sequences = []

    for i in range(0, len(frames) - seq_len + 1, step):

        seq = frames[i:i + seq_len]

        sequences.append(seq)

    return sequences

# POST-PROCESSING
def remove_consecutive_duplicates(words):
    result = []
    for w in words:
        if not result or result[-1] != w:
            result.append(w)
    return result

def remove_noise(words):
    noise_words = ['partikel-lah']
    return [w for w in words if w not in noise_words]

def keep_frequent_words(words, min_count=2):
    counter = Counter(words)
    return [w for w in words if counter[w] >= min_count]

def clean_sentence(words):
    # 1. hapus duplikat berurutan
    words = remove_consecutive_duplicates(words)

    # 2. hapus noise
    words = remove_noise(words)

    # 3. ambil kata yang sering muncul
    words = keep_frequent_words(words, min_count=2)

    # 4. hapus duplikat lagi
    words = remove_consecutive_duplicates(words)

    return words

# PREDICTION
def predict_sequences(sequences):
    results = []

    for seq in sequences:
        seq = np.array(seq)
        seq = np.expand_dims(seq, axis=0)

        pred = model.predict(seq, verbose=0)

        confidence = np.max(pred)
        label = labels[np.argmax(pred)]

        print(f"Prediksi: {label} | Confidence: {confidence:.2f}")

        results.append(label)

    return results

# MAIN PIPELINE
def main():
    # 1. extract frame
    frames = extract_frames(VIDEO_PATH)

    # 2. split jadi sequence
    sequences = split_sequences(frames, SEQ_LEN)
    print("Jumlah sequence:", len(sequences))

    if len(sequences) == 0:
        print("Tidak ada sequence yang valid!")
        return

    # 3. prediksi
    hasil_kata = predict_sequences(sequences)

    print("\nSebelum:", hasil_kata)

    # 4. clean hasil
    hasil_bersih = clean_sentence(hasil_kata)

    # 5. gabungkan kalimat
    kalimat = " ".join(hasil_bersih)

    print("\nSesudah:", kalimat)

# RUN
if __name__ == "__main__":
    main()