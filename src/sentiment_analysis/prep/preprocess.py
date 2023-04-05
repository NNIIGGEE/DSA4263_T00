### IMPORTS ###
import pandas as pd
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import os
import re

#%% READ IN RAW CSV FILE
path = "../datasets/reviews.csv" 
raw = pd.read_csv(path)


#%% DATA PREPROCESSING 

# Lowercase all words
raw['Text'] = raw['Text'].apply(str.lower)

# Remove stopwords using NLTK
stop_words = set(stopwords.words('english')) 
raw['no_stopwords'] = raw['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Tokenize text
tokenizer = TweetTokenizer()
raw['tokens'] = raw['no_stopwords'].apply(tokenizer.tokenize)

# Remove punctuation
punctuations = list(string.punctuation)
punctuations.append('...')

def remove_punctuation_word(word):
    '''
    Parameters
    ----------
    words : Takes in a single token.

    Returns
    -------
    new_list : Returns token without any special characters contained in it, excluding words with hyphen in between.

    '''
    lst = []
    corrected_word=""
    for i, char in enumerate(word):
        if (i!=0 and i!=(len(word)-1) and char == '-' and word[i+1] not in string.punctuation):
            corrected_word+=char
            
        elif char in string.punctuation:
            lst.append(corrected_word)
            corrected_word = ""
            continue
            
        else:
            corrected_word+=char
            
    lst.append(corrected_word)
    new_list = [x for x in lst if x != '']

    return new_list

remove_punctuation_word('made-in-the-usa')

def remove_punctuation(words):
    '''
    Parameters
    ----------
    words : Takes in the list of tokens of each customer review.

    Returns
    -------
    new_list : Returns a list of tokens with punctuations removed and tokens containing punctuation split.

    '''
    punctuations = list(string.punctuation)
    new_list = []
    for word in words:
        if word not in punctuations:
            new_word = remove_punctuation_word(word)
            new_list = new_list + new_word
    return new_list

raw['no_punc_1'] = raw['tokens'].apply(remove_punctuation)

# Remove short length words
def remove_short_words(words):
    '''
    Parameters
    ----------
    words : Takes in the list of tokens of each customer review.

    Returns
    -------
    list : Returns a list of tokens with short words removed.

    '''
    return [word for word in words if len(word) > 2]

raw['removed_short_word_2'] = raw['no_punc_1'].apply(remove_short_words)

# Convert amazon.com to amazon
def convert_strings(words):
    converted_words = []
    for word in words:
        if word == 'amazon.com':
            converted_words.append('amazon')
        else:
            converted_words.append(word)

    return converted_words

raw['converted_strings_3'] = raw['removed_short_word_2'].apply(convert_strings)

# Remove contractions
def remove_contractions(words):
    '''
    Parameters
    ----------
    words : Takes in the list of tokens of each customer review.

    Returns
    -------
    list : Returns a list of tokens with words containing contractions removed.

    '''
    contractions = ["'re", "'ve", "'d", "'m", "'ll", "n't"]
    new_words = []

    for word in words:
        # Check if the word is a contraction
        if any([word.endswith(c) for c in contractions]):
            continue
        else:
            # If it isn't a contraction, append the word to the new list as it is
            new_words.append(word)
    return new_words

raw['remove_contractions_4'] = raw['converted_strings_3'].apply(remove_contractions)

# Remove URLs
def remove_urls(words):
    '''
    Parameters
    ----------
    words : Takes in the list of tokens of each customer review.

    Returns
    -------
    list : Returns a list of tokens with URLs removed.

    '''
    # Define the regular expression pattern for URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    new_words = []

    for word in words:
        # Check if the word is a URL
        if not url_pattern.match(word):
            # If it isn't a URL, append the word to the new list
            new_words.append(word)

    return new_words

raw['remove_urls_5'] = raw['remove_contractions_4'].apply(remove_urls)

# Remove any remaining tokens containing digits or special characters
def remove_digits(words):
    '''
    Parameters
    ----------
    words : Takes in the list of tokens of each customer review.

    Returns
    -------
    list : Returns a list of tokens with tokens containing digits removed.

    '''
    pattern = re.compile(r'\d')
    clean_tokens = []

    for word in words:
        if not pattern.match(word):
            clean_tokens.append(word)
    
    return clean_tokens

raw['remove_digits_6'] = raw['remove_urls_5'].apply(remove_digits)

"""# Remove Typos
word_list = set(words.words()) # Load the 'words' corpus from NLTK

def remove_typos(tokens):
    '''
    Parameters
    ----------
    tokens : Takes in the list of tokens of each customer review.

    Returns
    -------
    clean_tokens : Returns a list of tokens with typos removed.

    '''
    clean_tokens = []
    for token in tokens:
        if token in word_list:
            clean_tokens.append(token)
    return clean_tokens

raw['remove_typos_7'] = raw['remove_digits_6'].apply(remove_typos)"""

# Lemmatization 
raw['pos_tags'] = raw['remove_digits_6'].apply(nltk.pos_tag) # POSTAG the tokens with NLTK for lemmatization

def get_wordnet_pos(treebank_tag):
    """
    Maps Penn Treebank POS tags to WordNet POS tags.

    Parameters
    ----------
    treebank_tag: (str or tuple): A Penn Treebank POS tag.

    Returns
    -------
    A WordNet POS tag.
    """
    if isinstance(treebank_tag, tuple):
        treebank_tag = treebank_tag[0]
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def lemmatize_tokens(col):
    """
    Lemmatizes a list of tokens based on their part-of-speech (POS) tags.

    Parameters
    ----------
    tokens (list): A list of tokens to be lemmatized.
    pos_tags (list): A list of POS tags corresponding to the tokens.

    Returns
    -------
    lemmas: A list of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()

    lemmas = []
    for token, pos_tag in col:
        wn_pos_tag = get_wordnet_pos(pos_tag)
        lemma = lemmatizer.lemmatize(token, pos=wn_pos_tag)
        lemmas.append(lemma)

    return lemmas

raw['lemmatized'] = raw.apply(lambda row: lemmatize_tokens(row['pos_tags']), axis=1)

# Remove duplicates created from lemmatization
def remove_duplicates(tokens):
    '''
    Parameters
    ----------
    tokens : Takes in the list of lemmatized tokens of each customer review.

    Returns
    -------
    A list of lemmatized tokens with no duplicates

    '''
    return list(set(tokens))

raw['cleaned'] = raw['lemmatized'].apply(remove_duplicates)

# %% OUTPUT TOKENS

final_df = raw[['Sentiment', 'Time', 'cleaned', 'Text']]
final_df.to_csv('../datasets/final_cleaned_tokens.csv')

