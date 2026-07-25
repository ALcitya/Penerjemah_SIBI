import os
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request
from src.predict_video import predict_video_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
ALLOWED_EXT = {".webm", ".mp4", ".avi", ".mov"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 *1024 *1024  # 50 MB

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", hasil="")

@app.route('/collection.html', methods=['GET'])
def collection():
    return render_template("collection.html")

@app.route('/contact.html', methods=['GET'])
def contact():
    return render_template("contact.html")


@app.route("/record_predict",methods=["POST"])
def record_predict():
    file =request.files.get("video")
    if not file:
        return jsonify({
        "prediction":
        "Video kosong"
        })
    original_name = secure_filename(file.filename) or "video.webm"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({
            "prediction": f"Format file tidak didukung. Hanya mendukung: {', '.join(ALLOWED_EXT)}"
        })
    
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    path=os.path.join(app.config["UPLOAD_FOLDER"],unique_name)
    file.save(path)
    print("Tersimpan", path)
    
    try:
        hasil = predict_video_file(path)
    except ValueError as e:
        print("Error prediksi:", e)
        hasil = "Tidak dapat mendeteksi gerakan tangan pada video"
    except Exception as e:
        print("Error tak terduga:", e)
        hasil = "Terjadi kesalahan saat memproses video"
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify({
        "prediction": hasil
    })

if __name__ == '__main__':
    app.run(debug=False)