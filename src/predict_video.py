import os
from pyparsing import warnings
# sembunyikan log tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# disable mediapipe warning
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from src.extract_frames import extract_frames
from src.translate_sentences import SentenceTranslator


# CONFIG
VIDEO_PATH = "./data/uploads/video_gerakan_2.mp4"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "sibi_model.keras")

# gunakan dataset hasil mediapipe
DATASET_PATH = "./data/hands"
SEQ_LEN = 20
STEP = 5
CONFIDENCE_THRESHOLD = 0.03

# LOAD MODEL & LABEL
model = load_model(model_path, compile=False)
labels = sorted([
    label for label in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, label))
])
# translator
translator = SentenceTranslator()
# SLIDING WINDOW
def sliding_window(frames, seq_len=20, step=5):
    sequences = []
    for i in range(0, len(frames) - seq_len + 1, step):
        seq = frames[i:i + seq_len]
        if len(seq) == seq_len:
            sequences.append(seq)
    return sequences

# POST PROCESSING
def remove_consecutive_duplicates(words):
    result = []
    seen = set()
    for w in words:
        if w not in seen:
            result.append(w)
            seen.add(w)
    return result

def keep_frequent_words(words, min_count=2):
    counter = Counter(words)
    return [
        w for w in words
        if counter[w] >= min_count
    ]
def remove_duplicates(words):

    result = []
    seen = set()

    for word in words:

        if word not in seen:

            result.append(word)
            seen.add(word)

    return result

def clean_sentence(words):

    # filter kata yang cukup sering muncul
    words = keep_frequent_words(
        words,
        min_count=2
    )

    # hapus kata berurutan
    words = remove_consecutive_duplicates(
        words
    )

    # hapus kata yang pernah muncul
    words = remove_duplicates(
        words
    )

    return words


# PREDICTION
def predict_sequences(sequences):
    results = []
    for i, seq in enumerate(sequences):
        seq = np.array(seq, dtype=np.float32)
        seq = seq / 255.0
        seq = np.expand_dims(seq, axis=0)
        pred = model.predict(seq, verbose=0)
        confidence = float(np.max(pred))
        label_index = int(np.argmax(pred))
        label = labels[label_index]
        print(
            f"Sequence {i+1} | "
            f"Prediksi: {label} | "
            f"Confidence: {confidence:.4f}"
        )
        # filter confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            results.append(label)
    return results

# MAIN PIPELINE
def main():
    # 1. extract hand frames
    frames = extract_frames(VIDEO_PATH)
    print(f"\nTotal frame tangan: {len(frames)}")
    if len(frames) < SEQ_LEN:
        print("Frame tidak cukup untuk sequence")
        return

    # 2. sliding window
    sequences = sliding_window(
        frames,
        seq_len=SEQ_LEN,
        step=STEP
    )
    print(f"Jumlah sequence: {len(sequences)}")
    if len(sequences) == 0:
        print("Tidak ada sequence valid")
        return

    # 3. predict
    hasil_kata = predict_sequences(sequences)
    if len(hasil_kata) == 0:
        print("\nTidak ada prediksi dengan confidence cukup")
        return
    # 4. clean sentence
    hasil_bersih = clean_sentence(hasil_kata)
    hasil_bersih = translator.combine_affixes(hasil_bersih)
    for word in hasil_bersih:
        translator.add_word(word)
    kalimat = translator.get_sentence()

    print("\nKalimat akhir:")
    print(kalimat)

def predict_video_file(video_path):
    frames = extract_frames(video_path)
    if len(frames) < SEQ_LEN:
        return "frame tidak cukup"
    sequences = sliding_window(frames, seq_len=SEQ_LEN, step=STEP)
    if len(sequences) == 0:
        return "tidak ada sequence valid"
    hasil_kata = predict_sequences(sequences)
    if len(hasil_kata) == 0:
        return "tidak ada prediksi"
    
    hasil_bersih = clean_sentence(hasil_kata)
    hasil_bersih = translator.combine_affixes(hasil_bersih)
    translator.reset()
    
    for word in hasil_bersih:
        translator.add_word(word)
        
    kalimat = " ".join(hasil_bersih)
    print("hasil_kata:", hasil_kata)

    hasil_bersih = clean_sentence(
    hasil_kata
    )

    print("hasil_bersih:", hasil_bersih)
    return kalimat

# RUN
if __name__ == "__main__":
    main()