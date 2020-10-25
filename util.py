import os
import re
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from variables import*

def get_class_names():
    return os.listdir(train_dir)

def load_image_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split
                                    )

    classes, images = load_data()
    train_datagen.fit(images)
    train_generator = train_datagen.flow(
                                        images, 
                                        images, 
                                        batch_size=batch_size_cnn
                                        )

    return train_generator, classes
    # train_generator = train_datagen.flow_from_directory(
    #                                 train_dir,
    #                                 target_size = target_size,
    #                                 color_mode = color_mode,
    #                                 batch_size = batch_size_cnn,
    #                                 classes = get_class_names(),
    #                                 shuffle = True)

    # validation_generator = train_datagen.flow_from_directory(
    #                                 train_dir,
    #                                 target_size = target_size,
    #                                 color_mode = color_mode,
    #                                 batch_size = batch_size_cnn,
    #                                 classes = get_class_names(),
    #                                 shuffle = True)

    # return train_generator, validation_generator

def load_data():
    if not os.path.exists(save_path):
        images = []
        classes = []
        url_strings = []
        dog_folders = os.listdir(train_dir)
        for label in list(dog_folders):
            label_dir = os.path.join(train_dir, label)
            label_images = []
            print('{} : {}'.format(label, len(os.listdir(label_dir))))
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)*rescale
                img = cv.resize(img, target_size)

                images.append(img)
                classes.append(label)

        images = np.array(images).astype('float32')
        classes = np.array(classes).astype('str')
        np.savez(save_path, name1=images, name2=classes)

    else:
        data = np.load(save_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']
    return classes, images

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df = df[['Discription', 'Breed']]
    df['Breed'] = df['Breed'].str.lower()
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
    
def load_text_data():
    df = preprocess_data(csv_path)
    classes = df['Breed'].values 

    encoder = LabelEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes)
    doggy_reviews = df['Discription'].values
    doggy_reviews = preprocessed_data(doggy_reviews)

    Ntest = int(val_split * len(classes))
    X, Y = shuffle(doggy_reviews, classes)
    Xtrain, Xtest = X[:-Ntest], X[-Ntest:]
    Ytrain, Ytest = Y[:-Ntest], Y[-Ntest:]
    return Xtrain, Xtest, Ytrain, Ytest

