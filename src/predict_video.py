import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import json
import gdown
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
from src.extract_frames import extract_frames
from src.translate_sentences import SentenceTranslator

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
labels_path = os.path.join(BASE_DIR, "models", "labels.json")

MODEL_DIR = os.environ.get("MODEL_DIR", "/data")  # fallback ke /data kalau env var gak diset
model_path = os.path.join(MODEL_DIR, "sibi_model.keras")
MODEL_FILE_ID = "10jZ4WS4_OT5EeUEqrOHVVxogrTx1IfWL"

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(model_path):
        print("Model belum ada di volume, downloading dari Google Drive...")
        gdown.download(id=MODEL_FILE_ID, output=model_path, quiet=False)
        print("Download model selesai.")
    else:
        print("Model sudah ada di volume, skip download.")

ensure_model()

SEQ_LEN = 20
DURASI_KATA = 3
OVERLAP_RATIO = 1/3
CONFIDENCE_THRESHOLD = 0.3

# LOAD MODEL & LABEL
model = load_model(model_path, compile=False)
with open(labels_path,"r", encoding="utf-8") as f:
    labels=json.load(f)
# translator
translator = SentenceTranslator()
def get_video_fps(video_path, default_fps=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default_fps
# SLIDING WINDOW
def sliding_window(frames, video_path, seq_len=SEQ_LEN, durasi_kata=DURASI_KATA, overlap_ratio=OVERLAP_RATIO):
    n = len(frames)
    fps = get_video_fps(video_path)
    window_size = max(seq_len, int(round(fps * durasi_kata)))
    step = max(1, int(window_size * overlap_ratio))
    sequences = []
    
    if n <= window_size:
        # video pendek
        idxs = np.linspace(0, n-1, seq_len).astype(int)
        sequences.append([frames[i] for i in idxs])
        return sequences
    for start in range(0, n - window_size+ 1, step):
        window = frames[start:start + window_size]
        # resample agar sama panjangnya
        idxs = np.linspace(0,len(window)-1, seq_len).astype(int)
        seq = [window[i] for i in idxs]
        sequences.append(seq)
    return sequences

def remove_consecutive_duplicates(words):
    result = []
    for w in words:
        if not result or result[-1] != w:
            result.append(w)
    return result

def keep_frequent_words(words, min_count=2):
    counter = Counter(words)
    affixes_prefix = ("awalan-", "akhiran-")
    return [
        w for w in words
        if counter[w] >= min_count or w.startswith(affixes_prefix)
    ]

def clean_sentence(words):
    if len(words) == 0:
        return words
    cleaned = keep_frequent_words(words, min_count=2)
    if len(cleaned) == 0:
        cleaned = words
    cleaned = remove_consecutive_duplicates(cleaned)
    return cleaned
# PREDICTION
def predict_sequences(sequences, batch_size=2):
    results = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        batch=np.array(chunk, dtype=np.float32)
        batch =batch[...,::-1] # convert bgr to rgb
        batch = batch / 255.0
        preds = model.predict(batch, verbose=0)
        
        for j, pred in enumerate(preds):
            confidence =float(np.max(pred))
            label_index = int(np.argmax(pred))
            label = labels[label_index]
            print(f"Sequence {i+j+1}: Prediksi: {label}, Confidence: {confidence:.2f}")
            if confidence >= CONFIDENCE_THRESHOLD:
                results.append(label)
                
        del batch
    return results

def predict_video_file(video_path):
    frames = extract_frames(video_path)
    if len(frames) < SEQ_LEN:
        return "frame tidak cukup"
    
    sequences = sliding_window(frames, video_path)
    print(f"Total sequences: {len(sequences)}")
    if len(sequences) == 0:
        return "tidak ada sequence valid"
    
    hasil_kata = predict_sequences(sequences)
    print("hasil_kata:", hasil_kata)
    if len(hasil_kata) == 0:
        return "tidak ada prediksi"
    
    hasil_bersih = clean_sentence(hasil_kata)
    print("hasil_bersih:", hasil_bersih)
    
    hasil_bersih = translator.combine_affixes(hasil_bersih)
    translator.reset()
    for word in hasil_bersih:
        translator.add_word(word)
        
    return translator.get_sentence()

# RUN
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_video_file(sys.argv[1]))
    else:
        print("Usage: python predict_video.py <video_path>")