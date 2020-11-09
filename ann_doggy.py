import os
import time
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
import pandas as pd
pd.options.mode.chained_assignment = None
logging.getLogger('tensorflow').disabled = True
import numpy as np
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from variables import *
np.random.seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def load_ann_data():
    df_all = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df = df_all[ann_cols+['Breed']]
    df = df.dropna(axis=0, how='any')
    df['Breed'] = df['Breed'].str.lower()
    df['Breed'] = df['Breed'].replace('afgan hound', 'afghan hound')
    df['Breed'] = df['Breed'].str.strip().values 

    df_cat = df[['Accomodation','Garden', 'Gender', 'Age', 'Size']]
    df[['Accomodation','Garden', 'Gender', 'Age', 'Size']] = label_encoding(df_cat)

    Y = df['Breed'].values
    del df['Breed']
    df = df.astype('int')
    X = df.values
    X, Y = shuffle(X, Y)

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    Ntest = int(0.15 * len(Y))
    Xtrain, Xtest = X[:-Ntest], X[-Ntest:]
    Ytrain, Ytest = Y[:-Ntest], Y[-Ntest:]

    return Xtrain, Xtest, Ytrain, Ytest

class ArtificialNNmodel:
    def __init__(self):
        if not os.path.exists(ann_weights):
            Xtrain, Xtest, Ytrain, Ytest = load_ann_data()
            self.Xtrain = Xtrain
            self.Xtest = Xtest
            self.Ytrain  = Ytrain
            self.Ytest  = Ytest
            self.size_output = len(set(self.Ytrain))
            self.n_features = int(self.Xtrain.shape[1])
            print(" No: of Classes : {}".format(self.size_output))

    def autoencoder_model(self):
        inputs = Input(shape=(self.n_features,), name='input_ann')
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.size_output, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        self.autoencoder_model()
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy'],
        )
        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=64,
                            epochs=30,
                            validation_data=[self.Xtest, self.Ytest]
                            )

    def save_model(self):
        self.model.save(ann_weights)

    def load_model(self):
        loaded_model = load_model(ann_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(0.001),
                        metrics=['accuracy']
                        )
        self.model = loaded_model

    def run(self):
        if os.path.exists(ann_weights):
            print(" ANN Model Loaded ")
            self.load_model()
        else:
            print(" ANN Model Training")
            self.train()
            self.save_model()
