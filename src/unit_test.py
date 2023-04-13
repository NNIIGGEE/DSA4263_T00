#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# test_sentiment_analysis.py
import unittest
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentiment_analysis.prep import preprocess
from nltk.corpus import wordnet

import pandas as pd
from sentiment_analysis.train.evaluate import get_sentiment, get_lstm_score, get_vader_score, convert_score, evaluate, select_best



class TestPreprocessing(unittest.TestCase):
    def test_remove_punctuation_word(self):
        self.assertEqual(preprocess.remove_punctuation_word("This is a sample text!"), ["This is a sample text"])
        
    def test_remove_stopwords(self):
        self.assertEqual(preprocess.remove_stopwords(["This", "is", "a", "sample", "text"]), ["This", "sample", "text"])
        
    def test_lemmatize_text(self):
        self.assertEqual(preprocess.lemmatize_text(["running", "cats"]), ["running", "cat"])
        
    def test_remove_punctuation(self):
        self.assertEqual(preprocess.remove_punctuation(["This", "is", "a", "sample", "text!"]), ["This", "is", "a", "sample", "text"])
        
    def test_remove_short_words(self):
        self.assertEqual(preprocess.remove_short_words(["This", "is", "a", "sample", "text"]), ["This", "sample", "text"])
        
    def test_convert_strings(self):
        self.assertEqual(preprocess.convert_strings(["amazon.com", "is", "a", "website"]), ["amazon", "is", "a", "website"])
        
    def test_remove_contractions(self):
        self.assertEqual(preprocess.remove_contractions(["I", "can't", "do", "it"]), ["I", "do", "it"])
        
    def test_remove_urls(self):
        self.assertEqual(preprocess.remove_urls(["This", "is", "a", "https://www.example.com", "link"]), ["This", "is", "a", "link"])
        
    def test_remove_digits(self):
        self.assertEqual(preprocess.remove_digits(["There", "are", "123", "digits", "in", "this", "text"]), ["There", "are", "digits", "in", "this", "text"])
        
    def test_get_wordnet_pos(self):
        self.assertEqual(preprocess.get_wordnet_pos('NN'), wordnet.NOUN)
        self.assertEqual(preprocess.get_wordnet_pos('JJ'), wordnet.ADJ)
        self.assertEqual(preprocess.get_wordnet_pos('VB'), wordnet.VERB)
        self.assertEqual(preprocess.get_wordnet_pos('RB'), wordnet.ADV)

    def test_lemmatize_tokens(self):
        col = [('cars', 'NNS'), ('are', 'VBP'), ('beautiful', 'JJ')]
        self.assertEqual(preprocess.lemmatize_tokens(col), ['car', 'be', 'beautiful'])

    def test_remove_duplicates(self):
        tokens = ['dog', 'cat', 'dog', 'horse', 'cat']
        self.assertEqual(preprocess.remove_duplicates(tokens), ['horse', 'dog', 'cat'])

        
class TestSentimentAnalysis(unittest.TestCase):
    
    def __init__(self, methodName: str):
        super().__init__(methodName=methodName)
        self.model = joblib.load('../trained_models/lstm_partial_SA.pkl')
        self.max_length = 100
        self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([""])
        
    def predict_sentiment(self, text):
        text_sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(text_sequence, maxlen=self.max_length)
        prediction = self.model.predict(padded_sequence)[0][0]
        if prediction >= 0.5:
            return "positive"
        else:
            return "negative"
        self.assertGreaterEqual(prediction, 0, msg="Value is less than 0")
        self.assertLessEqual(prediction, 1, msg="Value is greater than 1")
        
    # def test_positive_sentiment(self):
    #     text = "really enjoyed movie acting great plot engaging"
    #     sentiment = self.predict_sentiment(text)
    #     self.assertEqual(sentiment, "positive")
        
    # def test_negative_sentiment(self):
    #     text = "horrible hated it"
    #     sentiment = self.predict_sentiment(text)
    #     self.assertEqual(sentiment, "negative")
    
class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Create test data
        self.time = ['18/6/21', '7/7/21']
        self.texts = ['this is a very healthy dog food. good for their digestion. also good for small puppies. my dog eats her required amount at every feeding.'
                      , 'overall, this mix is okay, but, the sugar content is very, very high. you can actually see the sugar in the mix if you look closely. for pancakes or waffles, this is fine, but for savory concoctions, this is not acceptable. sausage balls, casseroles, and biscuits all come out almost inedible.this mix is also oddly gritty. the grit is very fine, but still makes my teeth want to curl.  i recommend pamela\'s baking and pancake mix, as it is smoother, and much more versatile, because it\'s not as sweet. a good cook knows you can always add more of something, but you can\'t take it out if you\'ve added too much.<br /><br />that being said, if you\'re traveling and this is the only mix they have, it\'s better than nothing.']
        self.tokens = [['puppy', 'also', 'eats', 'digestion', 'small', 'food', 'dog', 'require', 'amount', 'every', 'healthy', 'feeding', 'good']
                       ,['fine', 'closely', 'content', 'curl', 'say', 'bake', 'oddly', 'casseroles', 'biscuit', 'pancake', 'ball', 'still', 'come', 'see', 'good', 'mix', 'teeth', 'concoction', 'sugar', 'take', 'actually', 'waffle', 'grit', 'high', 'recommend', 'pamela', 'something', 'almost', 'overall', 'acceptable', 'also', 'want', 'nothing', 'smoother', 'sweet', 'inedible', 'add', 'always', 'savory', 'okay', 'sausage', 'pancakes', 'gritty', 'versatile', 'added', 'look', 'much', 'well', 'make', 'travel', 'cook', 'know']
]
        self.labels = [1, 0]
        self.x_train = pd.DataFrame({"Text":self.texts, "label":self.labels, "cleaned2":self.tokens, "Time":self.time})
        self.x_test = pd.DataFrame({"Text":self.texts, "label":self.labels, "cleaned2":self.tokens, "Time":self.time})
        
    def test_get_sentiment(self):
        # Test get_sentiment function
        self.assertEqual(get_sentiment(1), "positive")
        self.assertEqual(get_sentiment(0), "negative")

    def test_get_lstm_score(self):
        print("=====================TEST get_lstm_score()=====================")
        # Test get_lstm_score function
        output_df = get_lstm_score(self.x_train, self.x_test)
        # Check if the output dataframe has the right number of rows and columns
        self.assertEqual(output_df.shape[0], self.x_test.shape[0])
        self.assertEqual(output_df.shape[1], 4)
        # Check if the predicted sentiment is either 0 or 1
        self.assertTrue(all(x in [0, 1] for x in output_df['predicted'].values))

    def test_convert_score(self):
        # Test convert_score function
        polarity_scores = [{"compound":0.5}, {"compound":-0.5}]
        output = [convert_score(x) for x in polarity_scores]
        # Check if the output list has the right length
        self.assertEqual(len(output), len(polarity_scores))
        # Check if the predicted sentiment is either 0 or 1
        self.assertTrue(all(x in [0, 1] for x in output))

    def test_get_vader_score(self):
        print("=====================TEST get_vader_score()=====================")
        # Test get_vader_score function
        output_df = get_vader_score(self.x_test)
        # Check if the output dataframe has the right number of rows and columns
        self.assertEqual(output_df.shape[0], self.x_test.shape[0])
        self.assertEqual(output_df.shape[1], 5)
        # Check if the predicted sentiment is either 0 or 1
        self.assertTrue(all(x in [0, 1] for x in output_df['predicted'].values))

    def test_evaluate(self):
        # Test evaluate function
        y_test = np.array([1, 0, 0])
        predicted_df = pd.DataFrame({"predicted":[1, 0, 1]})
        output = evaluate(predicted_df, y_test)
        # Check if the output list has the right length
        self.assertEqual(len(output), 5)
        # Check if the output list has values between 0 and 1
        self.assertTrue(all(x >= 0 and x <= 1 for x in output))

    def test_select_best_vader(self):
        print("=====================TEST select_best(): Vader=====================")
        # Test select_best function
        results1 = [0.8, 0.6, 0.7, 0.7, 0.7]
        results2 = [0.7, 0.7, 0.8, 0.8, 0.7]
        output = select_best(results1, results2)

        self.assertEqual(output, 'Vader')
        
    def test_select_best_lstm(self):
        print("=====================TEST select_best(): LSTM=====================")
        # Test select_best function
        results2 = [0.8, 0.6, 0.7, 0.7, 0.7]
        results1 = [0.7, 0.7, 0.8, 0.8, 0.7]
        output = select_best(results1, results2)

        self.assertEqual(output, 'LSTM')
        
if __name__ == '__main__':
    unittest.main()
