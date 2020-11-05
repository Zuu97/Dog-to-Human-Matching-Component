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
        if not os.path.exists(rnn_weights):
            Xtrain_pad, Xtest_pad, Ytrain, Ytest = load_text_data()
            self.Xtrain_pad = Xtrain_pad
            self.Xtest_pad = Xtest_pad
            self.Ytrain  = Ytrain
            self.Ytest  = Ytest
            self.size_output = len(set(self.Ytest))

    def feature_extractor(self):
        inputs = Input(shape=(max_length,))
        x = Embedding(output_dim=embedding_dimS, input_dim=vocab_size, input_length=max_length, name='embedding')(inputs)
        x = Bidirectional(LSTM(size_lstm, unroll=True), name='bidirectional_lstm')(x)
        x = Dense(dense_1_rnn, activation='relu', name='dense1')(x)
        x = Dense(dense_1_rnn, activation='relu', name='dense2')(x)
        x = Dense(dense_2_rnn, activation='relu', name='dense3')(x)
        x = Dense(dense_2_rnn, activation='relu', name='dense4')(x)
        x = Dense(dense_3_rnn, activation='relu', name='dense5')(x)
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
                        optimizer=Adam(learning_rate), 
                        metrics=['accuracy']
                        )
        self.history = self.model.fit(
                                self.Xtrain_pad,
                                self.Ytrain,
                                batch_size=batch_size_rnn,
                                epochs=epochs_rnn,
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
