from flask import Flask, render_template, request 
import os
import sys
# tambahkan src ke path
sys.path.append("src")
from predict_video import predict_video_file

app = Flask(__name__)

upload_folder = 'uploads'
os.makedirs(upload_folder, exits_ok=True)
app.config['upload_folder'] = upload_folder

@app.route('/', methods=['GET','POST'])
def index():
    hasil = None
    if request.method == 'POST':
        if 'video' not in request.files:
            hasil = 'File video tidak ditemukan'
            return render_template('index.html', hasil=hasil)
        file = request.files['video']
        
        if file.filename == '':
            hasil = 'Belum memilih file video'
            return render_template('index.html', hasil=hasil)
        
        video_path = os.path.join(
            app.config['upload_folder'],
            file.filename
        )
        file.save(video_path)
        
        try:
            hasil = predict_video_file(video_path)
        except Exception as e:
            hasil = f"error :{str(e)}"
            
    return render_template('index.html', hasil=hasil)
if __name__ == '__main__':
    app.run(debug=True)