from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
# load model
model=load_model(
    "../models/sibi_model.keras",
    compile=False
)
# simpan visualisasi
plot_model(
    model,
    to_file="../image/cnn_lstm_model.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=100
)
print("visualisasi model berhasil disimpan")