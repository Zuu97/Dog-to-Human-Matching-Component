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
            self.cnn_model = DoggyCNN()
            self.rnn_model = DoggyRNN()
            self.cnn_model.run()
            self.rnn_model.run()

    def build_cnn_decoder(self):
        self.cnn_autoencoder = self.cnn_model.model
        inputs = Input(shape=(dense_1_rnn,))
        x = inputs
        for layer in self.cnn_autoencoder.layers[-18:]:
            layer.trainable = False
            x = layer(x)
        outputs = x
        self.cnn_decoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )

    def build_rnn_encoder(self):
        self.rnn_lstm= self.rnn_model.model
        self.rnn_lstm.summary()
        
    def dogMatcher(self):

        self.build_rnn_encoder()
        self.build_cnn_decoder()
        # self.cnn_autoencoder.summary()
        

    # def train(self):
    #     self.model.compile(
    #                       optimizer='Adam',
    #                       loss='categorical_crossentropy',
    #                       metrics=['accuracy']
    #                       )
    #     self.model.fit_generator(
    #                       self.train_generator,
    #                       steps_per_epoch = self.train_step,
    #                       validation_data = self.validation_generator,
    #                       validation_steps = self.validation_step,
    #                       epochs=epochs_cnn,
    #                       verbose=verbose
    #                     )

    # def save_model(self):
    #     print("Mobile Net TF Model Saving !")
    #     model_json = self.model.to_json()
    #     with open(cnn_architecture, "w") as json_file:
    #         json_file.write(model_json)
    #     self.model.save_weights(cnn_weights)

    # def load_model(self):
    #     K.clear_session() #clearing the keras session before load model
    #     json_file = open(cnn_architecture, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()

    #     self.model = model_from_json(loaded_model_json)
    #     self.model.load_weights(cnn_weights)

    #     self.model.compile(
    #                        loss='categorical_crossentropy', 
    #                        optimizer='Adam', 
    #                        metrics=['accuracy']
    #                        )
    #     print("Mobile Net TF Model Loaded !")


    # def run(self):
    #     if os.path.exists(cnn_weights):
    #         self.load_model()
    #     else:
    #         self.model_conversion()
    #         self.train()
    #         self.save_model()

model = DogToDogMatchingComponent()
model.dogMatcher()
