import pandas as pd
import nltk
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import string

# nltk.download('stopwords')


path = '/Users/admin/Downloads/DSA4263/reviews.csv'

raw = pd.read_csv(path)
data = pd.read_csv(path)

# lowercase words 

raw['Text'] = raw['Text'].apply(str.lower)

# remove stopwords using NLTK
stop_words = set(stopwords.words('english')) 
raw['no_stopwords'] = raw['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# tokenize words using NLTK TweetTokenizer
tokenizer = TweetTokenizer()
raw['tokens'] = raw['no_stopwords'].apply(tokenizer.tokenize)

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

data.to_csv('/Users/admin/Downloads/DSA4263/preprocessed_draft1.csv')

