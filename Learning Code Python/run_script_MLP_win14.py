# -*- coding: utf-8 -*-
import numpy as np
from main import main
import os

# # choose GPU (before importing Keras)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # # dynamically grow memory
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
# GPU Server 1 comment next two lines, uncomment third
# GPU Server 2 uncomment next two lines, comment third
from tensorflow.python.keras import backend as K

K.set_session(sess)
# tf.compat.v1.keras.backend.set_session(sess) # set this TensorFlow session as the default session for Keras


layer_dims = [[14, 8, 1], [14, 16, 1], [14, 32, 1], [14, 64, 1], [14, 128, 1],
              [14, 8, 8, 1], [14, 8, 16, 1], [14, 16, 8, 1], [14, 8, 32, 1], [14, 32, 8, 1], [14, 16, 32, 1],
              [14, 32, 16, 1], [14, 64, 8, 1], [14, 8, 64, 1], [14, 64, 16, 1], [14, 16, 64, 1], [14, 64, 32, 1],
              [14, 32, 64, 1], [14, 16, 16, 1], [14, 64, 64, 1], [14, 32, 32, 1], [14, 128, 128, 1], [14, 128, 8, 1],
              [14, 128, 16, 1], [14, 128, 32, 1], [14, 128, 64, 1], [14, 8, 128, 1], [14, 16, 128, 1], [14, 32, 128, 1],
              [14, 64, 128, 1], [14, 8, 8, 8, 1], [14, 8, 8, 16, 1], [14, 8, 8, 32, 1], [14, 8, 8, 64, 1],
              [14, 8, 16, 8, 1], [14, 8, 32, 8, 1], [14, 8, 64, 8, 1], [14, 16, 8, 8, 1], [14, 32, 8, 8, 1],
              [14, 64, 8, 8, 1], [14, 128, 8, 8, 1], [14, 8, 8, 128, 1], [14, 128, 64, 8, 1], [14, 64, 64, 8, 1],
              [14, 64, 128, 8, 1], [14, 16, 64, 8, 1], [14, 64, 16, 8, 1], [14, 128, 16, 8, 1], [14, 16, 128, 8, 1],
              [14, 32, 128, 8, 1], [14, 128, 32, 8, 1], [14, 64, 32, 8, 1], [14, 32, 64, 8, 1], [14, 16, 32, 8, 1],
              [14, 32, 16, 8, 1], [14, 16, 16, 8, 1], [14, 128, 128, 8, 1], [14, 32, 32, 8, 1], [14, 128, 64, 16, 1],
              [14, 64, 64, 16, 1], [14, 64, 128, 16, 1], [14, 16, 64, 16, 1], [14, 64, 16, 16, 1], [14, 128, 16, 16, 1],
              [14, 16, 128, 16, 1], [14, 32, 128, 16, 1], [14, 128, 32, 16, 1], [14, 64, 32, 16, 1],
              [14, 32, 64, 16, 1], [14, 16, 32, 16, 1], [14, 32, 16, 16, 1], [14, 128, 64, 32, 1], [14, 64, 64, 32, 1],
              [14, 64, 128, 32, 1], [14, 16, 64, 32, 1], [14, 64, 16, 32, 1], [14, 128, 16, 32, 1],
              [14, 16, 128, 32, 1], [14, 32, 128, 32, 1], [14, 128, 32, 32, 1], [14, 64, 32, 32, 1],
              [14, 32, 64, 32, 1], [14, 16, 32, 32, 1], [14, 32, 16, 32, 1], [14, 128, 64, 64, 1], [14, 64, 64, 64, 1],
              [14, 64, 128, 64, 1], [14, 16, 64, 64, 1], [14, 64, 16, 64, 1], [14, 128, 16, 64, 1],
              [14, 16, 128, 64, 1], [14, 32, 128, 64, 1], [14, 128, 32, 64, 1], [14, 64, 32, 64, 1],
              [14, 32, 64, 64, 1], [14, 16, 32, 64, 1], [14, 32, 16, 64, 1], [14, 128, 64, 128, 1],
              [14, 64, 64, 128, 1], [14, 64, 128, 128, 1], [14, 16, 64, 128, 1], [14, 64, 16, 128, 1],
              [14, 128, 16, 128, 1], [14, 16, 128, 128, 1], [14, 32, 128, 128, 1], [14, 128, 32, 128, 1],
              [14, 64, 32, 128, 1], [14, 32, 64, 128, 1], [14, 16, 32, 128, 1], [14, 32, 16, 128, 1]]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
num_epochs = [1, 3, 5]
l2_reg = [0, 0.0001, 0.001, 0.01, 0.1]
# minibatch_size = [1, 32, 64]
# memory_ = [1, 30, 90, 180, 240, 360]


for n in range(80):
    rate = np.random.choice(learning_rate)
    l = np.random.choice(layer_dims)
    e = np.random.choice(num_epochs)
    l2 = np.random.choice(l2_reg)
    # b = np.random.choice(minibatch_size)
    main({'method': 'MLP', 'data': 'original_beta1_normalized_interpolated_win14', 'n_repeats': 10, 'days_pred': 1,
          'memory': 1, 'layer_dims': l, 'learning_rate': rate, 'num_epochs': e, 'l2_reg': l2, 'minibatch_size': 1,
          'cnn_input': None, 'filters': None, 'kernel': None, 'pool': None, 'fc_units': None, 'output': None})

for n in range(80):
    rate = np.random.choice(learning_rate)
    l = np.random.choice(layer_dims)
    e = np.random.choice(num_epochs)
    l2 = np.random.choice(l2_reg)
    # b = np.random.choice(minibatch_size)
    main({'method': 'MLP', 'data': 'original_beta2_normalized_interpolated_win14', 'n_repeats': 10, 'days_pred': 1,
          'memory': 1, 'layer_dims': l, 'learning_rate': rate, 'num_epochs': e, 'l2_reg': l2, 'minibatch_size': 1,
          'cnn_input': None, 'filters': None, 'kernel': None, 'pool': None, 'fc_units': None, 'output': None})

for n in range(80):
    rate = np.random.choice(learning_rate)
    l = np.random.choice(layer_dims)
    e = np.random.choice(num_epochs)
    l2 = np.random.choice(l2_reg)
    # b = np.random.choice(minibatch_size)
    main({'method': 'MLP', 'data': 'original_beta3_normalized_interpolated_win14', 'n_repeats': 10, 'days_pred': 1,
          'memory': 1, 'layer_dims': l, 'learning_rate': rate, 'num_epochs': e, 'l2_reg': l2, 'minibatch_size': 1,
          'cnn_input': None, 'filters': None, 'kernel': None, 'pool': None, 'fc_units': None, 'output': None})

for n in range(80):
    rate = np.random.choice(learning_rate)
    l = np.random.choice(layer_dims)
    e = np.random.choice(num_epochs)
    l2 = np.random.choice(l2_reg)
    # b = np.random.choice(minibatch_size)
    main({'method': 'MLP', 'data': 'original_beta4_normalized_interpolated_win14', 'n_repeats': 10, 'days_pred': 1,
          'memory': 1, 'layer_dims': l, 'learning_rate': rate, 'num_epochs': e, 'l2_reg': l2, 'minibatch_size': 1,
          'cnn_input': None, 'filters': None, 'kernel': None, 'pool': None, 'fc_units': None, 'output': None})
