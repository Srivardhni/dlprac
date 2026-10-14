"""Experiment 7: Sentiment analysis using LSTM."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

def run():
    texts = ["I love this product!", "worst experience ever", "amazing service and quality", "really disappointed", "fantastic and inspiring", "I do not like this", "wonderful experience", "terrible waste of money", "had a great time", "ruined my day"]
    labels = [1,0,1,0,1,0,1,0,1,0]
    tok = Tokenizer(num_words=5000); tok.fit_on_texts(texts)
    X = pad_sequences(tok.texts_to_sequences(texts), maxlen=100)
    y = to_categorical(labels, 2)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = Sequential([Embedding(5000,128), LSTM(128), Dropout(0.5), Dense(2,activation="softmax")])
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    h = model.fit(X_train,y_train,epochs=25,batch_size=64,validation_data=(X_test,y_test),verbose=0)
    print("Test accuracy:", model.evaluate(X_test,y_test,verbose=0)[1])
    examples = ["I love this product!", "This is the worst experience"]
    p = np.argmax(model.predict(pad_sequences(tok.texts_to_sequences(examples),maxlen=100),verbose=0),axis=1)
    for text, label in zip(examples,p): print(text, "->", ["negative","positive"][label])
    for key in ("accuracy","loss"):
        plt.plot(h.history[key],label="Train"); plt.plot(h.history["val_"+key],label="Validation"); plt.legend(); plt.show()
