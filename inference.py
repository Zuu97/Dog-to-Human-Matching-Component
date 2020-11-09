import os
import re
import time
import cv2 as cv
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from variables import *

from model import ConvolutionalRecurrentModel
from tflite_converter import KerasToTFConversion
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img
    
def filter_images(image_labels):
    idxs = []
    for label in dog_classes:
        idx = np.where(image_labels==label)[0]
        if len(idx) > min_test_sample:
            idx = np.random.choice(idx, min_test_sample, replace=False)
        idxs.extend(idx.tolist())
    return idxs


def load_inference_data():
    if not os.path.exists(inference_save_path):
        print(" Inference Images are Saving ")
        df = pd.read_csv(csv_path, encoding='ISO 8859-1')
        df = df.drop_duplicates(subset=['ImageName'])
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')
        df['Breed'] = df['Breed'].str.lower()
        df['Breed'] = df['Breed'].replace('afgan hound', 'afghan hound')
        df['Breed'] = df['Breed'].str.strip().values 

        img_names = df['ImageName'].str.strip().values 

        image_labels = []
        inference_images = []
        url_paths = []
        dog_folders = os.listdir(train_dir)
        for label in list(dog_folders):
            label_dir = os.path.join(train_dir, label)
            for img_name in os.listdir(label_dir):
                img_ = img_name.split('.')[0].strip()
                if img_ not in img_names:
                    img_path = os.path.join(label_dir, img_name)
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img, target_size)
                    img = preprocessing_function(img)
                    inference_images.append(img)
                    image_labels.append(label)
                    url_paths.append(img_path)


        inference_images = np.array(inference_images).astype('float32')
        image_labels = np.array(image_labels).astype('str')
        image_urls = np.array(url_paths).astype('str')

        image_labels, inference_images, image_urls = shuffle(image_labels, inference_images, image_urls)
        idxs = filter_images(image_labels)

        inference_images = inference_images[idxs]
        image_labels = image_labels[idxs]
        image_urls = image_urls[idxs]

        np.savez(inference_save_path, name1=inference_images, name2=image_labels, name3=image_urls)

    else:
        print(" Inference Images are Loading ")
        data = np.load(inference_save_path, allow_pickle=True)
        inference_images = data['name1']
        image_labels = data['name2']
        image_urls = data['name3']

    return image_labels, inference_images, image_urls

def load_labeled_data(image_labels, inference_images, image_urls, label):
    idxs = (image_labels==label)
    labels = image_labels[idxs]
    images = inference_images[idxs]
    urls = image_urls[idxs]
    return labels, images, urls

class InferenceModel(object):
    def __init__(self):
        image_labels, inference_images, image_urls = load_inference_data()
        self.inference_images = inference_images
        self.image_labels = image_labels
        self.image_urls = image_urls
        crn_inference = KerasToTFConversion(crn_converter_path)
        cnn_inference = KerasToTFConversion(cnn_converter_path)

        if (not os.path.exists(cnn_converter_path)) or (not os.path.exists(crn_converter_path)):
            self.crn_model_obj = ConvolutionalRecurrentModel()
            self.crn_model_obj.run()

            if not os.path.exists(cnn_converter_path):
                ccn_model = self.crn_model_obj.cnn_encoder
                cnn_inference.TFconverter(ccn_model)
                print(" CNN keras model Converted to TensorflowLite")

            if not os.path.exists(crn_converter_path):
                crn_model = self.crn_model_obj.model
                crn_inference.TFconverter(crn_model)
                print(" CRN keras model Converted to TensorflowLite")

        cnn_inference.TFinterpreter()
        crn_inference.TFinterpreter()

        self.crn_inference = crn_inference
        self.cnn_inference = cnn_inference
        
    def extract_image_features(self, label):

        self.image_labels_class, self.inference_images_class, self.image_urls_class = load_labeled_data(
                                                                                        self.image_labels, 
                                                                                        self.inference_images, 
                                                                                        self.image_urls,
                                                                                        label)
        if not os.path.exists(n_neighbour_weights.format(label)):
            self.test_features = np.array(
                            [self.cnn_inference.Inference(img, True) for img in self.inference_images_class]
                                        )
            self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = n_neighbour
                                        )
            self.neighbor.fit(self.test_features)
            with open(n_neighbour_weights.format(label), 'wb') as file:
                pickle.dump(self.neighbor, file)
        else:
            with open(n_neighbour_weights.format(label), 'rb') as file:
                self.neighbor = pickle.load(file)

    def extract_text_features(self, text_pad, feature):
        return self.crn_inference.Inference([text_pad, feature])

    def predictions(self, text_pad, feature, show_fig=False):
        n_neighbours = {}
        fig=plt.figure(figsize=(8, 8))
        text_pad = self.extract_text_features(text_pad, feature)
        result = self.neighbor.kneighbors(text_pad)[1].squeeze()
        for i in range(n_neighbour):
            neighbour_img_id = result[i]
            img = self.inference_images_class[neighbour_img_id]
            url = self.image_urls_class[neighbour_img_id]
            # img = rescale_imgs(img)
            fig.add_subplot(1, 3, i+1)
            plt.title('Neighbour {}'.format(i+1))
            plt.imshow((img * 255).astype('uint8'))
            n_neighbours["Neighbour {}".format(i+1)] =  "{}".format(url)
        if show_fig:
            plt.show()
        return n_neighbours