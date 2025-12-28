import tensorflow as tf
import numpy as np
import os

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28*28)

os.makedirs("/data", exist_ok=True)
np.save("/data/x_train.npy", x_train)
np.save("/data/y_train.npy", y_train)

print("Preprocessing completed")
