import pandas as pd
from sentiment_analysis.prep import preprocess
from sentiment_analysis.train import evaluate, train_model
from topic_modelling.BERT import BERT_model
from topic_modelling.topic_modelling_LDA import full_LDA_topic_modelling


def RunModel(config):

    if config == 'train':

        print("=========================CLEAN DATA=========================")
        data = preprocess.clean_data("../datasets/reviews.csv", "training")
        x_train, x_test, y_train, y_test = preprocess.split_data(data)
        print("=========================DONE=========================")

        print("=========================TRAINING MODEL FOR COMPARISON=========================")
        train_model.run_lstm_training(x_train, x_test, y_train, y_test)
        print("=========================DONE=========================")

        print("=========================EVALUATING MODEL=========================")
        lstm_pred = evaluate.get_lstm_score(x_train,x_test)
        vader_pred = evaluate.get_vader_score(x_test)
        print("=========================DONE=========================")

        print("=========================LSTM=========================")
        evaluate.evaluate(lstm_pred, y_test)
        print("=========================VADER========================")
        evaluate.evaluate(vader_pred, y_test)

        print("=========================Output dataframes=========================")
        print(lstm_pred.head()) #.style
        print(vader_pred.head()) #.style
    
    elif config == 'test':
        """==== uses test data or smthng ===="""
    else:
        return "input not accepted"

def RunTopicModelling(config):
    df = pd.read_csv("../datasets/reviews.csv")
    data = preprocess.clean_data("../datasets/reviews.csv", "training")
    LDAModel = full_LDA_topic_modelling(data)
    BERTModel = BERT_model(df)
    return # result


print('model start') 
model_output = RunModel(config = 'train')

print('model done')


