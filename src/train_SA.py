from sentiment_analysis.prep import preprocess
from sentiment_analysis.train import evaluate, train_model

def run_sa_pipeline():
    print("=========================CLEANING DATA=========================")
    data = preprocess.clean_data("../datasets/reviews.csv", "training")
    x_train, x_test, y_train, y_test = preprocess.split_data(data)
    print("============================DONE============================")

    print("=========================TRAINING MODEL=========================")
    train_model.run_lstm_training(x_train, x_test, y_train, y_test)
    print("============================DONE============================")

    print("=========================GETTING PREDICTIONS=========================")
    lstm_pred = evaluate.get_lstm_score(x_train,x_test)
    vader_pred = evaluate.get_vader_score(x_test)
    print("=========================DONE=========================")

    print("======================EVALUATING MODELS======================")
    print("======================LSTM PERFORMANCE======================")
    lstm_res = evaluate.evaluate(lstm_pred, y_test)
    print("======================VADER PERFORMANCE======================")
    vader_res = evaluate.evaluate(vader_pred, y_test)
    print("============================SELECTING BEST MODEL============================")
    best_model = evaluate.select_best(lstm_res, vader_res)
    print("============================DONE============================")

    print("======================TRAINING SELECTED MODEL ON FULL DATA======================")
    if best_model == "LSTM":
        train_model.run_lstm_training_full(data)

    print("============================END OF PIPELINE============================")