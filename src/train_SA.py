from sentiment_analysis.prep import preprocess
from sentiment_analysis.train import evaluate, train_model
from sentiment_analysis.topic_modelling import topic_modelling_LDA

print("=========================CLEAN DATA=========================")
data = preprocess.clean_data("../datasets/reviews.csv", "training")
x_train, x_test, y_train, y_test = preprocess.split_data(data)
print("=========================DONE=========================")

print("=========================PERFORMING LDA=========================")
LDA_code = topic_modelling_LDA.full_LDA_topic_modelling(data)
print("=========================LDA DONE=========================")

print("=========================TRAINING MODEL FOR COMPARISON=========================")
train_model.run_lstm_training(x_train, x_test, y_train, y_test)
print("=========================DONE=========================")

print("=========================EVALUATING MODEL=========================")

print("=========================DONE=========================")

print("==========================TRAINING BEST MODELS (FULL DATA)==========================")
train_model.run_lstm_training_full(data)
print("==========================DONE==========================")