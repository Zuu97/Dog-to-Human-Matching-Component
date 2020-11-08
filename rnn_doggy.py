import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import csv
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from variables import *

import logging
logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df = df[['Discription', 'Breed']]
    df['Breed'] = df['Breed'].str.lower()
    df['Breed'] = df['Breed'].replace('afgan hound', 'afghan hound')
    df = df.dropna(axis=1, how='all') # drop columns which  
    df = df[df['Discription'].notna()]
    df = df.fillna(method='ffill')
    return df

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
    
def tokenizing_data(Xtrain, Xtest):
    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(num_words = 3000, oov_token="<OOV>")
        tokenizer.fit_on_texts(Xtrain)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

    Xtrain_seq = tokenizer.texts_to_sequences(Xtrain)
    Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=30, truncating='post')

    Xtest_seq  = tokenizer.texts_to_sequences(Xtest)
    Xtest_pad = pad_sequences(Xtest_seq, maxlen=30)
    return Xtrain_pad, Xtest_pad

def load_text_data():
    df = preprocess_data(csv_path)
    classes = df['Breed'].str.strip().values 

    encoder = LabelEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes)
    doggy_reviews = df['Discription'].values
    doggy_reviews = preprocessed_data(doggy_reviews)

    Ntest = int(0.15 * len(classes))
    X, Y = shuffle(doggy_reviews, classes)
    Xtrain, Xtest = X[:-Ntest], X[-Ntest:]
    Ytrain, Ytest = Y[:-Ntest], Y[-Ntest:]

    Xtrain_pad, Xtest_pad = tokenizing_data(Xtrain, Xtest)
    return Xtrain_pad, Xtest_pad, Ytrain, Ytest


class RecurrentNNmodel:
    def __init__(self):
        if not os.path.exists(rnn_weights):
            Xtrain_pad, Xtest_pad, Ytrain, Ytest = load_text_data()
            self.Xtrain_pad = Xtrain_pad
            self.Xtest_pad = Xtest_pad
            self.Ytrain  = Ytrain
            self.Ytest  = Ytest
            self.size_output = len(set(self.Ytest))

    def feature_extractor(self):
        inputs = Input(shape=(30,))
        x = Embedding(output_dim=512, input_dim=3000, input_length=30, name='embedding')(inputs)
        x = Bidirectional(LSTM(256, unroll=True), name='bidirectional_lstm')(x)
        x = Dense(512, activation='relu', name='dense1')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(256, activation='relu', name='dense3')(x)
        x = Dense(256, activation='relu', name='dense4')(x)
        x = Dense(64, activation='relu', name='dense5')(x)
        outputs = Dense(self.size_output, activation='softmax', name='dense_out')(x)

        model = Model(
                    inputs=inputs, 
                    outputs=outputs,
                    name='RNN_Model'
                    )
        self.model = model

        self.model.summary()

    def train(self):
        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer=Adam(0.0001), 
                        metrics=['accuracy']
                        )
        self.history = self.model.fit(
                                self.Xtrain_pad,
                                self.Ytrain,
                                batch_size=32,
                                epochs=10,
                                validation_data=(self.Xtest_pad,self.Ytest),
                                )

    def save_model(self):
        self.model.save(rnn_weights)
        print(" RNN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(rnn_weights)
        print(" RNN Model Loaded")

    def run(self):
        if os.path.exists(rnn_weights):
            self.load_model()
        else:
            self.feature_extractor()
            self.train()
            self.save_model()
