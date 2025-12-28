import numpy as np
import tensorflow as tf
import os

x = np.load("/data/x_train.npy")
y = np.load("/data/y_train.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x, y, epochs=5)

os.makedirs("/model", exist_ok=True)
model.export("/model")
