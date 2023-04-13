import pickle
from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
import pandas as pd
import redis
from sentiment_analysis.prep import preprocess
from train_TM import full_LDA_topic_modelling, BERT_model
import joblib
from sentiment_analysis.prep import preprocess
from topic_modelling.BERT import BERT_model 
from topic_modelling.topic_modelling_LDA import full_LDA_topic_modelling, visualise_sentiments, visualise_topics
from sentiment_analysis.train import evaluate

#source venv/bin/activate
import os
cwd = os.getcwd()
print(cwd)
path = cwd + '/sentiment_analysis/trained_models/lstm_full_SA.pkl'

from fastapi import APIRouter
from starlette.responses import JSONResponse

router = APIRouter()

@router.post('/classify_data')
def classify_data(csv_data: dict):
    
    output = evaluate.get_score(csv_data) #u need to pass in the physical object
    output_classifier = {output}
    return JSONResponse(output_classifier)

