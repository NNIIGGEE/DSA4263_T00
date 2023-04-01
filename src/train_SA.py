from sentiment_analysis.prep import preprocess
from sentiment_analysis.train import evaluate, train_model

data = preprocess.clean_data("../datasets/reviews.csv")
print("=========================CLEANED DATA=========================")

print("=========================TRAINING MODEL=========================")
train_model.run_training(data)
print("=========================TRAINING MODEL DONE=========================")