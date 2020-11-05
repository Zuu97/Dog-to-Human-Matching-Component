import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from util import load_rcn_data
from variables import *

from cnn import DoggyCNN
from rnn import DoggyRNN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DoggyRCN(object):
    def __init__(self):
        if not os.path.exists(rcn_weights):
            Xtrain_pad, Xtest_pad, Imgtrain, Imgtest = load_rcn_data()
            self.Xtrain_pad = Xtrain_pad
            self.Xtest_pad = Xtest_pad
            self.Imgtrain = Imgtrain
            self.Imgtest = Imgtest

            print(" Train image Shape : {}".format(Imgtrain.shape))
            print(" Test  image Shape : {}".format(Imgtest.shape))
            print(" Train review Shape: {}".format(Xtrain_pad.shape))
            print(" Test review Shape : {}".format(Xtest_pad.shape))
            
            self.rnn_model = DoggyRNN()
            self.cnn_model = DoggyCNN()
            self.rnn_model.run()
            self.cnn_model.run()

        self.build_rnn_encoder()
        self.build_cnn_encoder()
        self.image_extraction()
        
    def build_rnn_encoder(self):
        self.rnn_lstm= self.rnn_model.model
        inputs = self.rnn_lstm.input
        outputs = self.rnn_lstm.layers[-3].output
        self.rnn_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )
                            
    def build_cnn_encoder(self):
        self.cnn_mobilenet = self.cnn_model.model
        inputs = self.cnn_mobilenet.input
        outputs = self.cnn_mobilenet.layers[-2].output
        self.cnn_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )

    def image_extraction(self):
        self.Ytrain = self.cnn_encoder.predict(self.Imgtrain)
        self.Ytest  = self.cnn_encoder.predict(self.Imgtest)

    def MergedModel(self):
        input_rnn = self.rnn_encoder.input
        output_rnn = self.rnn_encoder(input_rnn)

        x = Dense(dense_1_rcn, activation='relu')(output_rnn)
        x = Dense(dense_1_rcn, activation='relu')(x)
        x = Dense(dense_1_rcn, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        output_cnn = Dense(dense_2_rcn)(x)

        self.model = Model(
                        inputs = input_rnn,
                        outputs = output_cnn,
                        name = 'RCN_Model'
                        )
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='mse'
                          )
        self.model.fit(
                        self.Xtrain_pad,
                        self.Ytrain,
                        validation_data=[self.Xtest_pad, self.Ytest],
                        batch_size = batch_size_rcn,
                        epochs=epochs_rcn,
                        verbose=verbose
                        )

    def save_model(self):
        self.model.save(rcn_weights)
        print(" RCN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(rcn_weights)
        print(" RCN Model Loaded")


    def run(self):
        if os.path.exists(rcn_weights):
            self.load_model()
        else:
            self.MergedModel()
            self.train()
            self.save_model()
