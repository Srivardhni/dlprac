"""Experiment 2: XOR using a small DNN."""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    model = Sequential([Dense(4, activation="relu", input_shape=(2,)), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=1000, verbose=0)
    print("Predictions:")
    print(np.round(model.predict(X, verbose=0)))
    model.summary()
