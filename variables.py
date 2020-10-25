import os
num_classes = 5
val_split = 0.15
verbose = 1

# CNN params
batch_size_cnn = 32
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
dense_3_cnn = 7 * 7 * 256
stride1 = (1,1)
stride2 = (2,2)
kernal_size = (5,5)
upconv1_dim = (7, 7, 256)
fs1 = 256
fs2 = 128
fs3 = 64
epochs_cnn = 20
train_dir = os.path.join(os.getcwd(), 'data/Images/')
save_path = os.path.join(os.getcwd(), 'data/npz_images.npz')
cnn_weights = "weights/cnn_autoencoder.h5"
cnn_architecture = "weights/cnn_autoencoder.json"

## RNN params
csv_path = 'dog_reviews.csv'
vocab_size = 3000
max_length = 30
embedding_dimS = 512
trunc_type = 'post'
oov_tok = "<OOV>"
epochs_rnn = 20
batch_size_rnn = 32
size_lstm  = 128
dense_1_rnn = 256
dense_2_rnn = 64
learning_rate = 0.0001
rnn_weights = "weights/dog_lstm.h5"
rnn_architecture = "weights/dog_lstm.json"

## Final Model Params 







final_model_weights = "weights/final_model.h5"
final_model_architecture = "weights/final_model.json"