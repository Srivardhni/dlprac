"""Experiment 5: Face CNN. Requires a face dataset organised in class folders."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def run():
    folder = Path(input("Path to face dataset folder (subfolder per person): ").strip())
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    data = tf.keras.utils.image_dataset_from_directory(folder, image_size=(64,64), color_mode="grayscale", batch_size=16, validation_split=0.2, subset="training", seed=42)
    model = Sequential([Conv2D(16,(3,3),activation="relu",input_shape=(64,64,1)), MaxPooling2D(), Conv2D(32,(3,3),activation="relu"), MaxPooling2D(), Flatten(), Dense(64,activation="relu"), Dense(len(data.class_names),activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(data,epochs=5)
    print("Person labels:", data.class_names)
