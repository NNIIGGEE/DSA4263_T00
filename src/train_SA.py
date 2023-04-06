from sentiment_analysis.prep import preprocess
from sentiment_analysis.train import evaluate, train_model
from sentiment_analysis.topic_modelling import topic_modelling_LDA

print("=========================CLEAN DATA=========================")
data = preprocess.clean_data("../datasets/reviews.csv")
print("=========================DONE=========================")

print("=========================PERFORMING LDA=========================")
LDA_code = topic_modelling_LDA.full_LDA_topic_modelling(data)
print("=========================LDA DONE=========================")

print("=========================TRAINING MODEL=========================")
train_model.run_training(data)
print("=========================TRAINING MODEL DONE=========================")
