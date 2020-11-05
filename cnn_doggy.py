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
from util import load_image_data, get_test_image
from variables import *
np.random.seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DoggyCNN(object):
    def __init__(self):
        if not os.path.exists(cnn_weights):
            train_generator, validation_generator = load_image_data()
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // batch_size_cnn
            self.validation_step = self.validation_generator.samples // valid_size_cnn

    def cnn_encoder_model(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    weights="imagenet",
                                                             )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = Dense(dense_1_cnn, activation='relu')(x)
        x = Dense(dense_1_cnn, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense_2_cnn, activation='relu')(x)
        x = Dense(dense_2_cnn, activation='relu')(x)
        x = Dense(dense_3_cnn, activation='relu')(x)
        x = Dense(dense_3_cnn, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(
                            inputs=inputs,
                            outputs=outputs,
                            name='CNN_Model'
                            )
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= self.train_step,
                          validation_data= self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs_cnn,
                          verbose=verbose
                        )

    def save_model(self):
        self.model.save(cnn_weights)
        print(" CNN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(cnn_weights)
        print(" CNN Model Loaded")

    def predictions(self, idx):
        img, label = get_test_image(idx)
        img = np.expand_dims(img, axis=0)
        Pimg = self.model.predict(img)
        Pimg = Pimg.reshape(*input_shape) 
        img = img.reshape(*input_shape) 

        fig=plt.figure(figsize=(8, 6))
        fig.add_subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(img)

        fig.add_subplot(1, 2, 2)
        plt.title('Predicted Image')
        plt.imshow(Pimg)

        plt.show()

    def run(self):
        if os.path.exists(cnn_weights):
            self.load_model()
        else:
            self.cnn_encoder_model()
            self.train()
            self.save_model()
