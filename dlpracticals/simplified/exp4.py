"""Experiment 4: Character CNN. Expects a local image dataset arranged by class folders."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def run():
    folder = Path(input("Path to character dataset folder: ").strip())
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    train = tf.keras.utils.image_dataset_from_directory(folder, image_size=(28,28), color_mode="grayscale", batch_size=32)
    model = Sequential([Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)), MaxPooling2D(), Flatten(), Dense(64, activation="relu"), Dropout(0.3), Dense(len(train.class_names), activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, epochs=5)
    print("Classes:", train.class_names)
