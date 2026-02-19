import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. load data
X = np.load('../data/processed/X_gray.npy')
Y = np.load('../data/processed/Y_gray.npy')
# konversi ke one-hot encoding
num_classes = len(np.unique(Y))
Y = tf.keras.utils.to_categorical(Y, num_classes)
# split data 8:2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 2. bangun model CNN-lSTM
model = Sequential([
    # TimeDistributed menerapkan Conv2D ke setiap 20 frame secara individu
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=X.shape[1:]),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Conv2D(64, (3,3), activation= 'relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),
    # lSTM untuk mempelajari gerakan
    LSTM(64, return_sequences=False),
    # fully connected layer
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 3. kompilasi model
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()
# 4. training model
history = model.fit(
    X_train, Y_train, 
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# 5. evaluasi dan penyimpanan model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"\nAkurasi Test:{test_acc: .4f}")

model.save('../models/sibi_model_gray.keras')