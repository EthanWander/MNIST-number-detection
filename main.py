import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


ASCII_CHARS = [".", ",", ":", ";", "+", "*", "?", "%", "$", "#", "@"]


def get_img_ascii(img):
    ascii_str = ""
    for line in img:
        for pixel in line:
            index = math.floor(len(ASCII_CHARS)*pixel/256)
            ascii_str += ASCII_CHARS[index]
        ascii_str += "\n"
    return ascii_str


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    y_train_categorical = keras.utils.to_categorical(y_train)

    x_train_norm = keras.utils.normalize(x_train, axis=1)
    x_test_norm = keras.utils.normalize(x_test, axis=1)

    model = keras.Sequential(
        [
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x_train_norm, y_train_categorical, batch_size=50, epochs=12, validation_split=0.2, verbose=1)

    model.save('model')

    predictions = model.predict([x_test_norm])

    print("\n\n")
    print("Choose pictures out of the 10,000 available\n")

    while True:
        print("Enter an arbitrary number between 0 and 9999: ")
        img_index = input()
        img_index = int(img_index)
        print("\n")
        print(get_img_ascii(x_test[img_index]))
        print(np.argmax(predictions[img_index]))
        print("\n\n")


if __name__ == "__main__":
    main()
