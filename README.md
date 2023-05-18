# DSA4263_T00
Final project DSA4263


## This project seeks to
Develop an end-end solution for Voice of Customer analysis to provide insights to end business users using a public customer feedback dataset. We applied differnt data sciencetools including sentiment analysis and topic modeling. The LSTM model is used to predict the sentiments while LDA model was implemented to generate the different topics

## See full project
Use preferred method to open VOC.ipynb file. The notebook runs the 2 training pipelines and display the insights gathered. It also has a function get_score that takes in new data in csv format and performs predictions on the data.

To get the predictions:
- Add file to repo, add in the path as the parameter to the function and simply run the cell. A csv file named "reviews_test_predictions_GROUPT00.csv" will be created.

## Use our API
Local deployment, as shown through the presentation over port=5001.

## Running training pipelines
### Sentiment Analysis training pipeline
- This pipeline performs training on LSTM and Vader model then compares the models' performance based on some metrics. Based on the performance of the models, the best model is selected and trained on full data.
```
cd src
python train_SA.py
```

### Topic Modeling training pipeline
- This pipeline trains the LDA and BERT model for topic modeling. Both model will generate scores and interpretation of the topics. Using human evaluation, we then selected LDA model and use it to get visualisations of the trends.
```
cd src
python train_TM.py
```
