from sentiment_analysis.prep import preprocess

data = preprocess.clean_data("datasets/reviews.csv")
print("=========================CLEANED DATA=========================")