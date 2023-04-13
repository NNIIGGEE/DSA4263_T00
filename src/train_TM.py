from sentiment_analysis.prep import preprocess
from topic_modelling.BERT import BERT_model 
from topic_modelling.topic_modelling_LDA import full_LDA_topic_modelling

def run_TP_pipeline():
    print("=========================CLEANING DATA=========================")
    data = preprocess.clean_data("../datasets/reviews.csv", "training")
    print("============================DONE============================")

    print("=========================LDA MODEL=========================")
    LDA_data = full_LDA_topic_modelling(data)
    print("============================DONE============================")

    try:
        print("=========================BERT MODEL=========================")
        BERT_data = BERT_model(data)
        print("============================DONE============================")

    except:
        print("=========================BERT MODEL FAILED DUE TO IMPORTS=========================")

    print("============================END OF PIPELINE============================")

