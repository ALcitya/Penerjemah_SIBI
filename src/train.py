import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load data
X = np.load('../data/prep/grayscale/X_gray.npy')
Y = np.load('../data/prep/grayscale/Y_gray.npy')

# Normalisasi
X = X.astype('float32') / 255.0

# Jika Y masih berupa label (bukan one-hot)
if len(Y.shape) > 1:
    labels = np.argmax(Y, axis=1)
else:
    labels = Y

# Distribusi kelas
unique, counts = np.unique(labels, return_counts=True)

# Jumlah kelas
num_classes = len(unique)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# One-hot encoding setelah split
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

print("Shape input untuk model:", X.shape[1:])

# 2. Bangun model CNN-LSTM
model = Sequential([
    
    TimeDistributed(
        Conv2D(32, (3,3), activation='relu'),
        input_shape=X.shape[1:]
    ),
    TimeDistributed(MaxPooling2D((2,2))),
    
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    
    TimeDistributed(Flatten()),
    
    # LSTM untuk sequence frame
    LSTM(64, return_sequences=False),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Training
history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# 5. Evaluasi
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"\nAkurasi Test: {test_acc*100:.4f}%")

# 6. Plot hasil training
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# 7. Simpan model
model.save('../models/sibi_model_gray.keras')