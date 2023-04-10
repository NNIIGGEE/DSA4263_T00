import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def run_lstm_training(x_train, x_test, y_train, y_test):
    '''
    Trains LSTM model on partial dataset for comparison

    Parameters
    ----------
    x_train, x_test, y_train, y_test: cleaned data that has been split

    Returns
    -------
    Save trained model
    '''

    # Tokenize data
    tokenizer = Tokenizer(num_words=10971) #10971 is the number of unique tokens in our dataset
    tokenizer.fit_on_texts(x_train['cleaned2'])
    sequences = tokenizer.texts_to_sequences(x_train['cleaned2'])
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # Convert test set to sequences using the same tokenizer object
    sequences2 = tokenizer.texts_to_sequences(x_test['cleaned2'])
    padded_sequences2 = pad_sequences(sequences2, maxlen=100)

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10971, 64),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, y_train, epochs=10, batch_size=128, validation_data=(padded_sequences2, y_test))
    filename = "lstm_partial_SA.pkl"
    path = "../trained_models/" + filename
    joblib.dump(model, path)

    print("You can find trained model in trained_models")


def run_lstm_training_full(data):
    '''
    Trains LSTM model with full dataset

    Parameters
    ----------
    data : Takes in cleaned dataframe, data should have 5 columns -> label, time, cleaned, text and cleaned2

    Returns
    -------
    Save trained model
    '''

    # Tokenize data
    tokenizer = Tokenizer(num_words=10971)
    tokenizer.fit_on_texts(data['cleaned2'])
    sequences = tokenizer.texts_to_sequences(data['cleaned2'])
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10971, 64),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, data['label'], epochs=10, batch_size=128, validation_split=0.2)
    filename = "lstm_full_SA.pkl"
    path = "../trained_models/" + filename
    joblib.dump(model, path)

    print("You can find trained model in trained_models")








    