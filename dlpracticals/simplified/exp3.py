"""Experiment 3: MNIST digit classification."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

def run():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    model = Sequential([Flatten(input_shape=(28, 28)), Dense(128, activation="relu"), Dense(10, activation="softmax")])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Test accuracy:", acc, "Test loss:", loss)
    print("Predicted:", int(np.argmax(model.predict(X_test[:1], verbose=0))))
    print("Actual:", int(y_test[0]))
    for key, title in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        plt.plot(history.history[key], label="Train")
        plt.plot(history.history["val_" + key], label="Validation")
        plt.title(title); plt.legend(); plt.show()
