from flask import Flask, jsonify, render_template, request
import os

from src.predict_video import predict_video_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", hasil="")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("video")
    if not file:
        return render_template(
            "index.html",
            hasil="Video tidak ditemukan"
        )
    path = os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
    file.save(path)
    
    hasil = predict_video_file(path)
    return render_template("index.html", hasil=hasil)

@app.route("/record_predict",methods=["POST"])
def record_predict():
    file =request.files.get("video")
    if not file:
        return jsonify({
        "prediction":
        "Video kosong"
        })
    path=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
    file.save(path)
    print("Tersimpan", path)
    hasil=predict_video_file(path)
    response={
    "prediction":hasil}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)