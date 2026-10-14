"""Experiment 9: Tiny English-to-Tamil encoder-decoder structure demo.
This demonstrates the architecture only; it is not a useful trained translator."""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def run():
    # Toy token IDs: English input and Tamil target sequences.
    encoder_input = Input(shape=(3,))
    enc_emb = Embedding(20,16)(encoder_input)
    _, h, c = LSTM(32,return_state=True)(enc_emb)
    decoder_input = Input(shape=(3,))
    dec_emb = Embedding(20,16)(decoder_input)
    dec = LSTM(32,return_sequences=True)(dec_emb,initial_state=[h,c])
    output = Dense(20,activation="softmax")(dec)
    model = Model([encoder_input,decoder_input],output)
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    X = np.array([[1,2,3],[4,5,6]])
    Y = np.array([[1,2,3],[4,5,6]])[...,None]
    model.fit([X,X],Y,epochs=2,verbose=0)
    print("Encoder-decoder model created and trained on toy token IDs.")
