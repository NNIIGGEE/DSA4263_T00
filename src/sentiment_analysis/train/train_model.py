import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def run_training(data):
    # Tokenize data
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['cleaned2'])
    sequences = tokenizer.texts_to_sequences(data['cleaned2'])
    padded_sequences = pad_sequences(sequences)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(padded_sequences, data['label'], test_size=0.2, random_state = 42)

    # Build RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(2000, 64),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
    filename = "lstm_SA.pkl"
    path = "../trained_models/" + filename
    joblib.dump(model, path)

    print("Model Saved")








    