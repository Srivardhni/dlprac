# Handwritten exam keyword guide

Use these as the experiment-specific terms to include in your own explanation and code. Follow the actual lab manual and your teacher's rubric.

1. **Perceptron:** input matrix, target, weights, bias, learning rate, weighted sum/dot product, threshold/step activation, prediction, error, weight update, decision boundary.
2. **XOR with DNN:** XOR truth table, `Sequential`, `Dense`, hidden layer, ReLU, sigmoid, binary cross-entropy, Adam, `fit`, `predict`.
3. **MNIST:** `mnist.load_data()`, normalize by 255, `to_categorical`, `Flatten`, `Dense`, softmax, categorical cross-entropy, epochs, evaluate, accuracy/loss curves.
4. **Character CNN:** image loading, grayscale, resize 28×28, normalize, label encoding/one-hot, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, softmax, prediction vs actual.
5. **Face CNN:** OpenCV, `CascadeClassifier`, `detectMultiScale`, face crop/resize, normalization, `LabelEncoder`, `Conv2D`, max pooling, softmax, test evaluation, confusion matrix/classification report.
6. **Language RNN:** corpus, `Tokenizer`, word index, input sequences, padding, `Embedding`, `SimpleRNN`, Dense softmax, next-word prediction.
7. **Sentiment LSTM:** text labels, `Tokenizer`, `texts_to_sequences`, `pad_sequences`, `Embedding`, `LSTM`, `Dropout`, softmax, categorical cross-entropy, sentiment prediction.
8. **POS tagging Seq2Seq:** word/tag vocabularies, integer sequences, padding, encoder, decoder, embeddings, LSTM states, `return_sequences=True`, sparse categorical cross-entropy, POS tag prediction.
9. **English→Tamil translation:** source/target tokenizers, start/end tokens if used, encoder-decoder, embeddings, LSTM, teacher forcing/shifted target, softmax, decode predicted token IDs.
10. **GAN image generation:** generator, discriminator, latent/random noise, Dense, LeakyReLU, BatchNormalization, reshape 28×28×1, tanh, sigmoid, binary cross-entropy, alternate discriminator/generator training.

## Common Keras skeleton
```python
model = Sequential([
    # experiment-specific layers
])
model.compile(optimizer="adam", loss="...", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=...)
loss, accuracy = model.evaluate(X_test, y_test)
```

This skeleton does not replace the experiment-specific algorithm. Experiment 1 uses a manually updated perceptron; experiments 8–10 need architecture-specific logic.
