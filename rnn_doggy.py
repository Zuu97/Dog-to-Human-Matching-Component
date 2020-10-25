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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout
from variables import *
from util import load_text_data

import logging
logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DoggyRNN:
    def __init__(self):
        if not (os.path.exists(rnn_architecture)  and os.path.exists(rnn_weights)):
            print('FUck')
            Xtrain, Xtest, Ytrain, Ytest = load_text_data()
            self.Xtrain = Xtrain
            self.Ytrain = Ytrain
            self.Xtest  = Xtest
            self.Ytest  = Ytest
            print("Train input shape : {}".format(Xtrain.shape))
            print("Test  input shape : {}".format(Xtest.shape))
            self.size_output = len(set(self.Ytest))
        
    def tokenizing_data(self):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(self.Xtrain)

        Xtrain_seq = tokenizer.texts_to_sequences(self.Xtrain)
        self.Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=max_length, truncating=trunc_type)

        # print(Counter([len(x) for x in Xtrain_seq]))

        Xtest_seq  = tokenizer.texts_to_sequences(self.Xtest)
        self.Xtest_pad = pad_sequences(Xtest_seq, maxlen=max_length)
        self.tokenizer = tokenizer

    def feature_extractor(self):
        inputs = Input(shape=(max_length,))
        x = Embedding(output_dim=embedding_dimS, input_dim=vocab_size, input_length=max_length, name='embedding')(inputs)
        x = Bidirectional(LSTM(size_lstm), name='bidirectional_lstm')(x)
        x = Dense(dense_1_rnn, activation='relu', name='dense1')(x)
        x = Dense(dense_1_rnn, activation='relu', name='dense2')(x)
        x = Dense(dense_2_rnn, activation='relu', name='dense3')(x)
        x = Dense(dense_2_rnn, activation='relu', name='dense4')(x)
        x = Dense(dense_2_rnn, activation='relu', name='dense5')(x)
        outputs = Dense(self.size_output, activation='sigmoid', name='dense_out')(x)

        model = Model(inputs=inputs, outputs=outputs)
        self.model = model

    def train(self):
        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer=Adam(learning_rate), 
                        metrics=['accuracy']
                        )
        self.model.summary()
        self.history = self.model.fit(
                                self.Xtrain_pad,
                                self.Ytrain,
                                batch_size=batch_size_rnn,
                                epochs=epochs_rnn,
                                validation_data=(self.Xtest_pad,self.Ytest),
                                )

    def save_model(self):
        print("RNN LSTM Model Saving !")
        model_json = self.model.to_json()
        with open(rnn_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(rnn_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(rnn_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(rnn_weights)

        self.model.compile(
                           loss='sparse_categorical_crossentropy', 
                           optimizer=Adam(learning_rate), 
                           metrics=['accuracy']
                           )
        print("RNN LSTM Model Loaded !")

    def run(self):
        if os.path.exists(rnn_weights):
            self.load_model()
        else:
            self.tokenizing_data()
            self.feature_extractor()
            self.train()
            # self.save_model()

if __name__ == "__main__":
    model = DoggyRNN()
    model.run()
