from flask import Flask, request, render_template
from io import BytesIO
import pandas as pd
from src.sentiment_analysis.train import evaluate


app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def handle_newfile():
    if request.method == "POST":
        file = request.files['file']
        predictions = evaluate.test_get_score(BytesIO(file.read()))
        htmlcode = (predictions.to_html())
        textfile = open('templates/displayresult.html', 'w')
        textfile.write(htmlcode)
        textfile.close()
        return render_template('displayresult.html')

    else:
        return render_template('uploadfile.html')
