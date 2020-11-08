import os
import re
import time
import cv2 as cv
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import pickle
import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from variables import *

from cnn_doggy import ConvolutionalNNmodel
from rnn_doggy import RecurrentNNmodel
from ann_doggy import ArtificialNNmodel

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

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def load_filtered_images(img_names):
    images = []
    containing_img_names = []
    dog_folders = os.listdir(train_dir)
    for label in list(dog_folders):
        label_dir = os.path.join(train_dir, label)
        for img_name in os.listdir(label_dir):
            img_ = img_name.split('.')[0].strip()
            if img_ in img_names:
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, target_size)
                img = preprocessing_function(img)
                images.append(img)
                if img_ not in containing_img_names:
                   containing_img_names.append(img_) 

    images = np.array(images).astype('float32')
    return images, containing_img_names

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

def load_crn_data():
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df = df.drop_duplicates(subset=['ImageName'])
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')
    df['Breed'] = df['Breed'].str.lower()
    df['Breed'] = df['Breed'].replace('afgan hound', 'afghan hound')
    df['Breed'] = df['Breed'].str.strip().values 

    img_names = df['ImageName'].str.strip().values 
    images, containing_img_names = load_filtered_images(img_names)
    df = df.loc[df['ImageName'].isin(containing_img_names)]
    doggy_reviews = df['Discription'].values
    doggy_reviews = preprocessed_data(doggy_reviews)

    df_cat = df[['Accomodation','Garden', 'Gender', 'Age', 'Size']]
    df[['Accomodation','Garden', 'Gender', 'Age', 'Size']] = label_encoding(df_cat)

    features = df[ann_cols].values

    Ntest = int(0.15 * len(doggy_reviews))
    features, reviews, images = shuffle(features, doggy_reviews, images)
    Reviewtrain, Reviewval = reviews[:-Ntest], reviews[-Ntest:]
    Xtrain_pad, Xtest_pad = tokenizing_data(Reviewtrain, Reviewval)
    Imgtrain, Imgtest = images[:-Ntest], images[-Ntest:]
    Anntrain, Anntest = features[:-Ntest], features[-Ntest:]
    return Anntrain, Anntest, Xtrain_pad, Xtest_pad, Imgtrain, Imgtest

# load_crn_data()
class ConvolutionalRecurrentModel(object):
    def __init__(self):
        if not os.path.exists(crn_weights):
            Anntrain, Anntest, Xtrain_pad, Xtest_pad, Imgtrain, Imgtest = load_crn_data()
            self.Xtrain_pad = Xtrain_pad
            self.Xtest_pad = Xtest_pad
            self.Imgtrain = Imgtrain
            self.Imgtest = Imgtest
            self.Anntrain = Anntrain
            self.Anntest = Anntest

            print(" Train image Shape : {}".format(Imgtrain.shape))
            print(" Test  image Shape : {}".format(Imgtest.shape))
            print(" Train review Shape: {}".format(Xtrain_pad.shape))
            print(" Test review Shape : {}".format(Xtest_pad.shape))
            print(" Train feature Shape: {}".format(Anntrain.shape))
            print(" Test feature Shape : {}\n".format(Anntest.shape))
            
            self.rnn_model = RecurrentNNmodel()
            self.cnn_model = ConvolutionalNNmodel()
            self.ann_model = ArtificialNNmodel()
            self.rnn_model.run()
            self.cnn_model.run()
            self.ann_model.run()

        self.build_rnn_encoder()
        self.build_cnn_encoder()
        self.build_ann_encoder()
        self.image_extraction()

    def build_ann_encoder(self):
        self.rnn_dense= self.ann_model.model
        inputs = self.rnn_dense.input
        outputs = self.rnn_dense.layers[7].output
        self.ann_encoder = Model(
                            inputs = inputs,
                            outputs = outputs
                            )

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
        input_ann = self.ann_encoder.input

        output_rnn = self.rnn_encoder(input_rnn)
        output_ann = self.ann_encoder(input_ann)

        merged = Concatenate(axis=1)([output_rnn, output_ann])

        x = Dense(256, activation='relu')(merged)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_cnn = Dense(64)(x)

        self.model = Model(
                        inputs = [input_rnn, input_ann],
                        outputs = output_cnn,
                        name = 'crn_Model'
                        )
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='mse'
                          )
        self.model.fit(
                        [self.Xtrain_pad, self.Anntrain],
                        self.Ytrain,
                        validation_data=[
                                        [self.Xtest_pad, self.Anntest], 
                                        self.Ytest
                                        ],
                        batch_size = 16,
                        epochs=30
                        )

    def save_model(self):
        self.model.save(crn_weights)
        print(" CRN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(crn_weights)
        print(" CRN Model Loaded")


    def run(self):
        if os.path.exists(crn_weights):
            self.load_model()
        else:
            self.MergedModel()
            self.train()
            self.save_model()

model = ConvolutionalRecurrentModel()
model.run()
