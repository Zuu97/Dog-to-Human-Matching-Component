import os
import json
import pandas as pd
import numpy as np
from variables import *
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from inference import InferenceModel
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.preprocessing.image import img_to_array

import requests
from PIL import Image
from util import *
from flask import Flask
from flask import jsonify
from flask import request
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
'''
        python -W ignore app.py
'''
def tokenize_inference_text(X):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=30, truncating='post')
    return X_pad
def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def label_encoding(df_cat):
    if not os.path.exists(encoder_dict_path):
        encoder_dict = defaultdict(LabelEncoder)
        encoder = df_cat.apply(lambda x: encoder_dict[x.name].fit_transform(x))
        encoder.apply(lambda x: encoder_dict[x.name].inverse_transform(x))
        with open(encoder_dict_path, 'wb') as handle:
            pickle.dump(encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(encoder_dict_path, 'rb') as handle:
            encoder_dict = pickle.load(handle)
    return df_cat.apply(lambda x: encoder_dict[x.name].transform(x))

def get_feature_data(feature_data):
    feature_data = eval(feature_data)
    df_dict = {col:[feature_value] for col, feature_value in zip(ann_cols, feature_data)}
    df = pd.DataFrame(df_dict,
                      columns=ann_cols)
    df_cat = df[['Accomodation','Garden', 'Gender', 'Age', 'Size']]
    df[['Accomodation','Garden', 'Gender', 'Age', 'Size']] = label_encoding(df_cat)
    return df.values.squeeze()

def get_prediction_data(data):
    text, label, feature_data = data["text"], data["label"], data['feature']
    feature = get_feature_data(feature_data)
    label = str(label).lower()
    text = preprocessed_data(text)
    text_pad = tokenize_inference_text(text)[0]
    text_pad = text_pad.reshape(1, -1)
    feature = feature.reshape(1, -1)
    return text_pad, feature, label

model = InferenceModel()

app = Flask(__name__)

# message = {
#     "text" : "I would like to arrange a playdate for my female small size Maltese puppy it is very playful,active and have a good behaviour with other pets and behave well with strangers love go for walks. we live in kalutara.",
#     "label" : "shih tzu",
#     "feature" : "['flat', 'no', 6, 'femmale', 'adult', 'medium']"
#     }

@app.route("/predict", methods=["GET", "POST"])
def predict(show_fig=False):
    message = request.get_json(force=True)
    if len(message) == 3:
        text_pad, feature, label = get_prediction_data(message)
        model.extract_image_features(label)
        n_neighbours = model.predictions(text_pad, feature, show_fig)
        response = {
            "neighbours": n_neighbours
                    }
        return jsonify(response)
    else:
        return "Please input both Breed and the text content"

if __name__ == "__main__": 
    app.run(debug=True, host='0.0.0.0', port= 5000, threaded=False, use_reloader=False)
