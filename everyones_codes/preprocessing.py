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
test = raw['cleaned']
data['cleaned'] = raw['cleaned']

data.to_csv('/Users/admin/Downloads/DSA4263/preprocessed_draft1.csv')

