#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# data preprocessing
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import string

path = '/Users/admin/Downloads/DSA4263/reviews.csv'
# path = './datasets/reviews.csv'

raw = pd.read_csv(path)
data = pd.read_csv(path)

# lowercase words 

raw['Text'] = raw['Text'].apply(str.lower)

# remove stopwords using NLTK
stop_words = set(stopwords.words('english')) 
raw['no_stopwords'] = raw['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# tokenize words using NLTK TweetTokenizer
tokenizer = TweetTokenizer()
raw['tokens'] = raw['no_stopwords'].apply(word_tokenize)

# lemmatization 
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

raw['lemmatized'] = raw['tokens'].apply(lemmatize_text)


# remove punctuation
punctuations = list(string.punctuation)

def remove_punctuation(list):
    new_list = []
    for word in list:
        if word not in punctuations:
            new_list.append(word)
    return new_list

raw['cleaned'] = raw['lemmatized'].apply(remove_punctuation)

#remove short length words
def remove_short_words(words):
    return [word for word in words if len(word) > 2]
raw['removed_short_word'] = raw['cleaned'].apply(remove_short_words)

#convert amazon.com to amazon
def convert_strings(words):
    converted_words = []
    for word in words:
        if word == 'amazon.com':
            converted_words.append('amazon')
        else:
            converted_words.append(word)

    return converted_words

raw['converted_strings'] = raw['removed_short_word'].apply(convert_strings)

#remove contractions
def remove_contractions(words):
    contractions = ["'re", "'ve", "'d", "'m", "'ll", "n't"]
    new_words = []

    for word in words:
        # Check if the word is a contraction
        if any([word.endswith(c) for c in contractions]):
            continue
        else:
            # If it isn't a contraction, append the word to the new list as is
            new_words.append(word)
    return new_words

raw['remove_contractions'] = raw['converted_strings'].apply(remove_contractions)

import re

def remove_urls(words):
    # Define the regular expression pattern for URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    new_words = []

    for word in words:
        # Check if the word is a URL
        if not url_pattern.match(word):
            # If it isn't a URL, append the word to the new list
            new_words.append(word)

    return new_words
raw['remove_urls'] = raw['remove_contractions'].apply(remove_urls)

def remove_digits(words):
    new_words = []
    for word in words:
        if not any(char.isdigit() for char in word):
            new_words.append(word)

    return new_words


raw['remove_digits'] = raw['remove_urls'].apply(remove_digits)

def remove_ellipses(words):
    new_words = []
    for word in words:
        if "…" not in word:
            new_words.append(word)
    return new_words


raw['remove_ellipses'] = raw['remove_digits'].apply(remove_ellipses)

data['cleaned'] = raw['remove_ellipses']

data['cleaned'] = data['cleaned'].apply(' '.join)

# Tokenize data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['cleaned'])
sequences = tokenizer.texts_to_sequences(data['cleaned'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Label encoding
labels = np.array(data['Sentiment'].replace({'positive': 1, 'negative': 0}))

# train test split
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42)

# Build RNN model
tf.random.set_seed(1234)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')

# Get predictions
y_pred = model.predict(x_test)

# Compute ROC curve and area under the curve (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Compute precision, recall, and F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred.round(), average='binary')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# Print precision, recall, and F1 score
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1_score))







