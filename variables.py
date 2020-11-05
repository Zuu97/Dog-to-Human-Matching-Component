import os
num_classes = 5
val_split = 0.15
verbose = 1
seed = 42

# CNN params
batch_size_cnn = 12
valid_size_cnn = 6
color_mode = 'rgb'
width = 224
height = 224
n_channels = 3
target_size = (width, height)
input_shape = (width, height, n_channels)
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
dense_1_cnn = 512
dense_2_cnn = 256
dense_3_cnn = 64
stride1 = (1,1)
stride2 = (2,2)
kernal_size = (5,5)
upconv1_dim = (7, 7, 256)
fs1 = 256
fs2 = 128
fs3 = 64
epochs_cnn = 5
train_dir = os.path.join(os.getcwd(), 'data/Images/')
save_path = os.path.join(os.getcwd(), 'weights/numpy_images.npz')
cnn_weights = "weights/cnn_autoencoder.h5"
cnn_converter_path = "weights/cnn_model.tflite"

## RNN params
csv_path = 'data/dog_reviews.csv'
vocab_size = 3000
max_length = 30
embedding_dimS = 512
trunc_type = 'post'
oov_tok = "<OOV>"
epochs_rnn = 10
batch_size_rnn = 32
size_lstm  = 256
dense_1_rnn = 512
dense_2_rnn = 256 
dense_3_rnn = 64
learning_rate = 0.0001
rnn_weights = "weights/rnn_classifier.h5"

## RCN Model Params 
tokenizer_path = 'weights/tokenizer.pickle'
epochs_rcn = 30
dense_1_rcn = 128 
dense_2_rcn = 64
batch_size_rcn = 32
keep_prob = 0.3
rcn_weights = "weights/rcn_model.h5"
rcn_converter_path = "weights/rcn_model.tflite"

##Inference
host = '0.0.0.0'
port = 5000
inference_save_path = os.path.join(os.getcwd(), 'weights/inference_images.npz')
dog_classes = {'shih tzu', 'papillon', 'maltese', 'afghan hound', 'beagle'}
n_neighbour_weights = 'weights/nearest neighbor weight folder/nearest neighbour {}.pkl'
n_neighbour = 3
min_test_sample = 30
