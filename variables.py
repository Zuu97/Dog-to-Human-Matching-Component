import os
seed = 1234

# CNN params
color_mode = 'rgb'
target_size = (224, 224)
input_shape = (224, 224, 3)

train_dir = 'data/Images/'
save_path = 'weights/numpy_images.npz'
cnn_weights = "weights/cnn_weights.h5"
cnn_converter_path = "weights/cnn_model.tflite"

# ANN params
ann_cols = ['Accomodation','Garden','Hours', 'Gender', 'Age', 'Size']
encoder_dict_path = 'weights/encoder dict.pickle'
ann_weights = "weights/ann_weights.h5"
keep_prob = 0.5

## RNN params
csv_path = 'data/data.csv'
rnn_weights = "weights/rnn_weights.h5"

## crn Model Params 
tokenizer_path = 'weights/tokenizer.pickle'
crn_weights = "weights/crn_weights.h5"
crn_converter_path = "weights/crn_model.tflite"

##Inference
inference_save_path = os.path.join(os.getcwd(), 'weights/inference_images.npz')
dog_classes = {'shih tzu', 'papillon', 'maltese', 'afghan hound', 'beagle'}
n_neighbour_weights = 'weights/nearest neighbor weight folder/nearest neighbour {}.pkl'
n_neighbour = 3
min_test_sample = 30
