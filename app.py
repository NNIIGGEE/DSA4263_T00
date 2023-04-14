from flask import Flask, request, render_template
from io import BytesIO
import pandas as pd
from src.sentiment_analysis.train import evaluate, train_model


app = Flask(__name__)

@app.route('/train', methods=['GET','POST'])
def handle_trainfile():
    print("function start")
    if request.method == "POST":
        file = request.files['file']
        print(file)
        print("html reuqest input")
        predictions = train_model.run_lstm_training_full(BytesIO(file.read()))

        return render_template('uploadfile_copy.html')

    else:
        return render_template('uploadfile_copy.html')

@app.route('/predict', methods=['GET','POST'])
def handle_newfile():
    print("function start")
    if request.method == "POST":
        file = request.files['file']
        print("html reuqest input")
        predictions = evaluate.test_get_score(BytesIO(file.read()))
        print("html evaluated")
        htmlcode = (predictions.to_html())
        print("html open")
        textfile = open('templates/displayresult.html', 'w')
        textfile.write(htmlcode)
        textfile.close()
        return render_template('displayresult.html')

    else:
        return render_template('uploadfile.html')

if __name__ == "__main__":
     app.run(debug=True ,port=5001,use_reloader=False) 