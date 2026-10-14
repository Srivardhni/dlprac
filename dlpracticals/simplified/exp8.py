"""Experiment 8: Small Seq2Seq POS-tagging demonstration."""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def run():
    words = {"<pad>":0,"the":1,"cat":2,"runs":3}
    tags = {"<pad>":0,"DET":1,"NOUN":2,"VERB":3}
    X = np.array([[1,2,3],[1,2,0],[2,3,0]])
    y = np.array([[1,2,3],[1,2,0],[2,3,0]])
    enc_in = Input(shape=(3,))
    _, h, c = LSTM(16, return_state=True)(Embedding(4,8)(enc_in))
    dec_in = Input(shape=(3,))
    dec = LSTM(16, return_sequences=True)(Embedding(4,8)(dec_in), initial_state=[h,c])
    out = Dense(4,activation="softmax")(dec)
    model = Model([enc_in,dec_in],out)
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit([X,X],y[...,None],epochs=5,verbose=0)
    print("POS predictions:", np.argmax(model.predict([X[:1],X[:1]],verbose=0),axis=-1))
