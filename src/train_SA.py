from sentiment_analysis.prep import preprocess
# from sentiment_analysis.train import evaluate, train_model

print("=========================CLEAN DATA=========================")
data = preprocess.clean_data("../datasets/reviews.csv")
print("=========================DONE=========================")


# print("=========================TRAINING MODEL=========================")
# train_model.run_lstm_training_full(data)
# print("=========================DONE=========================")