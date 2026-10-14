"""Experiment 10: Minimal GAN training on MNIST."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten

def run():
    (X, _), _ = tf.keras.datasets.mnist.load_data()
    X = (X.astype("float32") - 127.5) / 127.5
    X = X.reshape(-1,28,28,1)[:5000]
    latent = 100
    generator = Sequential([Dense(128,input_dim=latent),LeakyReLU(negative_slope=0.2),BatchNormalization(),Dense(784,activation="tanh"),Reshape((28,28,1))])
    discriminator = Sequential([Flatten(input_shape=(28,28,1)),Dense(128),LeakyReLU(negative_slope=0.2),Dense(1,activation="sigmoid")])
    discriminator.compile(optimizer="adam",loss="binary_crossentropy")
    discriminator.trainable = False
    gan = Sequential([generator,discriminator])
    gan.compile(optimizer="adam",loss="binary_crossentropy")
    discriminator.trainable = True
    for _ in range(100):
        idx = np.random.randint(0,len(X),32)
        real = X[idx]
        noise = np.random.normal(0,1,(32,latent))
        fake = generator.predict(noise,verbose=0)
        discriminator.train_on_batch(real,np.ones((32,1)))
        discriminator.train_on_batch(fake,np.zeros((32,1)))
        gan.train_on_batch(np.random.normal(0,1,(32,latent)),np.ones((32,1)))
    samples = generator.predict(np.random.normal(0,1,(9,latent)),verbose=0)
    for i, img in enumerate(samples):
        plt.subplot(3,3,i+1); plt.imshow(img[:,:,0],cmap="gray"); plt.axis("off")
    plt.show()
