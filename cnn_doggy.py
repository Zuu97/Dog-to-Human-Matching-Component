import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from variables import *
np.random.seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_class_names():
    return os.listdir(train_dir)

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def load_image_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range = 20,
                                    shear_range = 0.2,
                                    zoom_range = 0.15,
                                    width_shift_range= 0.20,
                                    height_shift_range= 0.20,
                                    horizontal_flip = True,
                                    validation_split= 0.15,
                                    preprocessing_function=preprocessing_function
                                    )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = 12,
                                    classes = get_class_names(),
                                    subset = 'training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = 6,
                                    classes = get_class_names(),
                                    subset = 'validation',
                                    shuffle = True)

    return train_generator, validation_generator

class ConvolutionalNNmodel(object):
    def __init__(self):
        if not os.path.exists(cnn_weights):
            train_generator, validation_generator = load_image_data()
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // 12
            self.validation_step = self.validation_generator.samples // 6

    def cnn_encoder_model(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    weights="imagenet",
                                                             )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(5, activation='softmax')(x)
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
                          epochs=3
                        )

    def save_model(self):
        self.model.save(cnn_weights)
        print(" CNN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(cnn_weights)
        print(" CNN Model Loaded")

    def run(self):
        if os.path.exists(cnn_weights):
            self.load_model()
        else:
            self.cnn_encoder_model()
            self.train()
            self.save_model()
