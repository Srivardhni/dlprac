"""Experiment 6: Simple next-word prediction with an RNN."""
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def run():
    texts = ["deep learning is useful", "deep learning uses neural networks", "neural networks learn patterns"]
    tok = Tokenizer()
    tok.fit_on_texts(texts)
    seqs = []
    for line in texts:
        ids = tok.texts_to_sequences([line])[0]
        seqs += [ids[:i+1] for i in range(1, len(ids))]
    data = pad_sequences(seqs)
    X, y = data[:, :-1], data[:, -1]
    model = Sequential([Embedding(len(tok.word_index)+1, 16), SimpleRNN(32), Dense(len(tok.word_index)+1, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=100, verbose=0)
    print("Word index:", tok.word_index)
    print("Training accuracy:", model.evaluate(X, y, verbose=0)[1])
