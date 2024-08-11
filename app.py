from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from scipy.special import softmax
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

scaler = StandardScaler()
sia = SentimentIntensityAnalyzer()
with open('randomForestModelRoberta1.pkl', 'rb') as file:
    roberta = pickle.load(file)

with open('randomForestModelRVaders.pkl', 'rb') as file:
    vader = pickle.load(file)

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("model.pickle", "rb") as f:
    model = pickle.load(f)

def calculate_sentiment_scores(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = softmax(outputs.logits, axis=1).flatten().tolist()

    # Return sentiment scores as a list
    return probabilities

def DataF(rs):
    rs = [0.8655868172645569, 0.12227807939052582, 0.012135038152337074]
    testing = {
    'robertneg': [rs[0]],
    'robertnut': [rs[1]],
    'robertnpos': [rs[2]]
    }
    return pd.DataFrame(testing)

def get_sentiment_label(probabilities):
    labels = ["Negative", "Neutral", "Positive"]
    max_index = probabilities.index(max(probabilities))
    return labels[max_index]

def vader_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score > -0.05 and compound_score < 0.05:
        return "Neutral"
    else:
        return "Negative"

def calculate_compound_roberta(t):
    weight_neg = -1
    weight_neut = 0
    weight_pos = 1
    compound_score = (t[0] * weight_neg) + (t[1] * weight_neut) + (t[2] * weight_pos)
    return compound_score

def get(robertaLabel):
    if robertaLabel == 'Negative':
        rp = random.randint(1, 2)
    elif robertaLabel == 'Positive':
        rp = random.randint(4, 5)
    else:
        rp = 3
    return rp

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    return render_template("./index.html")

@app.route('/output', methods=['GET', 'POST'])
def predict():
    rp = 0
    vp = 0
    text = request.form['review']
    vaders_scores = sia.polarity_scores(text)
    roberta_scores = calculate_sentiment_scores(text)
    inp = [[calculate_compound_roberta(roberta_scores)]]
    roberta_predict = roberta.predict(pd.DataFrame(inp))
    vp = vader.predict([[vaders_scores['compound']]])
    robertaLabel = get_sentiment_label(roberta_scores)
    vaderLabel = vader_sentiment_label(vaders_scores['compound'])
    rp = get(robertaLabel)
    vp = get(vaderLabel)
    
    return render_template("./output.html", text = text,vaders = vaders_scores,roberta = roberta_scores,rp =rp,vp = vp,robertaLabel = get_sentiment_label(roberta_scores),vaderLabel = vader_sentiment_label(vaders_scores['compound']),name = request.form['name'])

if __name__=='__main__':
    app.run()