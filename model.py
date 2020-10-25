import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from util import load_image_data
from variables import *

from cnn import DoggyCNN
from rnn import DoggyRNN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DogToDogMatchingComponent(object):
    def __init__(self):
        if not (os.path.exists(final_model_weights)  and os.path.exists(final_model_architecture)):
            self.rnn_model = DoggyRNN()
            self.cnn_model = DoggyCNN()
            self.rnn_model.run()
            self.cnn_model.run()

    def build_rnn_encoder(self):
        self.rnn_lstm= self.rnn_model.model
        inputs = Input(shape=(max_length,))
        x = inputs
        for layer in self.rnn_lstm.layers[1:-2]:
            # layer.trainable = False
            x = layer(x)
        outputs = x
        self.rnn_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )
                            
    def build_cnn_decoder(self):
        self.cnn_autoencoder = self.cnn_model.model
        inputs = Input(shape=(dense_2_rnn,))
        x = inputs
        for layer in self.cnn_autoencoder.layers[-18:]:
            layer.trainable = False
            x = layer(x)
        outputs = x
        self.cnn_decoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )

    def dogMatcher(self):
        self.build_rnn_encoder()
        self.build_cnn_decoder()

        inputs = Input(shape=(max_length,))
        rnn_pred = self.rnn_encoder(inputs)
        outputs = self.cnn_decoder(rnn_pred)

        self.model = Model(
                        inputs = inputs,
                        outputs = outputs,
                        name = 'Dog to Dog Matching Component'
                        )
        self.model.summary()


    def train(self):

        self.model.compile(
                          optimizer='Adam',
                          loss='mse'
                          )
        self.model.fit(
                        self.train_generator,
                        steps_per_epoch = self.train_step,
                        epochs=epochs_cnn,
                        verbose=verbose
                        )

        # self.model.compile(
        #                   optimizer='Adam',
        #                   loss='categorical_crossentropy',
        #                   metrics=['accuracy']
        #                   )
        # self.model.fit_generator(
        #                   self.train_generator,
        #                   steps_per_epoch = self.train_step,
        #                   validation_data = self.validation_generator,
        #                   validation_steps = self.validation_step,
        #                   epochs=epochs_cnn,
        #                   verbose=verbose
        #                 )

    def save_model(self):
        print("RCN Model Saving !")
        model_json = self.model.to_json()
        with open(final_model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(final_model_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(final_model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(final_model_weights)

        self.model.compile(
                          optimizer='Adam',
                          loss='mse'
                          )
        print("RCN Model Loaded !")


    # def run(self):
    #     if os.path.exists(cnn_weights):
    #         self.load_model()
    #     else:
    #         self.model_conversion()
    #         self.train()
    #         self.save_model()

model = DogToDogMatchingComponent()
model.dogMatcher()
