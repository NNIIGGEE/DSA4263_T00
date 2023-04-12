from sentiment_analysis.prep import preprocess
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, classification_report, f1_score, recall_score

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
    tokenizer = Tokenizer(num_words=10971)
    tokenizer.fit_on_texts(cleaned['cleaned2'])
    sequences = tokenizer.texts_to_sequences(cleaned['cleaned2'])
    padded_sequences = pad_sequences(sequences, max_len=100)

    # Load model and predict
    model = joblib.load("../trained_models/lstm_full_SA.pkl")
    predicted = model.predict(padded_sequences)
    cleaned['predicted'] = predicted.round()
    cleaned['predicted_sentiment'] = cleaned["predicted"].apply(lambda row: get_sentiment(row))
    cleaned['predicted_sentiment_probability'] = predicted

    final_df = cleaned[['Text', 'Time', 'predicted_sentiment_probability', 'predicted_sentiment']]
    final_df.to_csv("../reviews_test_predictions_T00.csv")
    print("Predicted csv has been generated")

def get_lstm_score(x_train, x_test):
    '''
    Gets lstm score for sentiment analysis

    Parameters
    ----------
    x_test : data to get predictions

    Returns
    -------
    dataframe with predicted values
    '''
    
    tokenizer = Tokenizer(num_words=10971) #10971 is the number of unique tokens in our dataset
    tokenizer.fit_on_texts(x_train['cleaned2'])
    sequences = tokenizer.texts_to_sequences(x_train['cleaned2'])
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # Convert test set to sequences using the same tokenizer object
    sequences2 = tokenizer.texts_to_sequences(x_test['cleaned2'])
    padded_sequences2 = pad_sequences(sequences2, maxlen=100)

    model = joblib.load("../trained_models/lstm_partial_SA.pkl")
    predicted = model.predict(padded_sequences2)
    final_df = x_test[['Time', 'Text', 'cleaned2']]
    final_df['predicted'] = predicted.round()
    print("=======================LSTM PREDICTIONS========================")
    print(final_df[['Text','predicted']].head())
    return final_df

def convert_score(row):
    '''
    Convert vader polarity score

    Parameters
    ----------
    row : predicted polarity score

    Returns
    -------
    Either value 1 or 0 that stands for positive and negative sentiment
    '''
    compound = row["compound"]
    if compound >= 0: 
        score = 1
    else:
        score = 0
    return score

def get_vader_score(x_test):
    '''
    Gets vader score for sentiment analysis

    Parameters
    ----------
    x_test : data to get predictions

    Returns
    -------
    dataframe with predicted values
    '''
    model = SentimentIntensityAnalyzer()
    final_df = x_test[['Time', 'Text', 'cleaned2']]
    final_df["predicted_polarity"] = final_df["cleaned2"].apply(lambda x: model.polarity_scores(x))
    final_df["predicted"] = final_df["predicted_polarity"].apply(lambda row: convert_score(row))
    print("=======================VADER PREDICTIONS========================")
    print(final_df[['Text','predicted']].head())
    return final_df

def evaluate(predicted, y_test):
    '''
    Gets model performance for sentiment analysis

    Parameters
    ----------
    predicted : dataframe with predicted value
    y_test : true value

    Returns
    -------
    prints out the different metrics and get results
    '''
    results = []
    y_pred = predicted["predicted"]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    results.append(accuracy)
    results.append(precision)
    results.append(auc)
    results.append(f1)
    results.append(recall)

    print("Accuracy = ",accuracy)
    print("Precision = ", precision)
    print("ROC AUC = ", auc)
    print(classification_report(y_test, y_pred))
    return results

def select_best(results1, results2):
    '''
    Selects the best model based on various metric

    Parameters
    ----------
    results1, results2 : list of the metrics of the different model from evaluation, both lists should be equal length

    Returns
    -------
    Shows which model is selected
    '''
    metrics = ['accuracy', 'precision', 'ROC AUC', 'Recall', 'F1 score']
    count1 = 0
    count2 = 0
    for i in range(len(results1)):
        metric = metrics[i]
        if results1[i] > results2[i]:
            count1 += 1
            print("LSTM performed better in", metric)
        else:
            count2 += 1
            print("Vader performed better in", metric)
    
    if count1 >= count2:
        print("LSTM is the better model")
        return "LSTM"
    else:
        print("Vader is the better model")
        return "Vader"
    
