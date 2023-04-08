from sentiment_analysis.prep import preprocess
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_sentiment(val):
    '''
    Convert numerical prediction to text

    Parameters
    ----------
    val : predicted value

    Returns
    -------
    either "positive" or "negative" sentiment
    '''
    if val == 1.0:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment

def get_score(file):
    '''
    Gets prediction for sentiment analysis

    Parameters
    ----------
    file : path of file with testing data

    Returns
    -------
    CSV file with predicted sentiment probabilities
    '''

    # Prep testing data
    cleaned = preprocess.clean_data(file, "testing")
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(cleaned['cleaned2'])
    sequences = tokenizer.texts_to_sequences(cleaned['cleaned2'])
    padded_sequences = pad_sequences(sequences)

    # Load model and predict
    model = joblib.load("../trained_models/lstm_full_SA.pkl")
    predicted = model.predict(padded_sequences)
    cleaned['predicted'] = predicted.round()
    cleaned['predicted_sentiment'] = cleaned["predicted"].apply(lambda row: get_sentiment(row))
    cleaned['predicted_sentiment_probability'] = predicted

    final_df = cleaned[['Text', 'Time', 'predicted_sentiment_probability', 'predicted_sentiment']]
    final_df.to_csv("../reviews_test_predictions_T00.csv")
    print("Predicted csv has been generated")
