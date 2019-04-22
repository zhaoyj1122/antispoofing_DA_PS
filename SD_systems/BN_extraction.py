import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model, Model
from keras.callbacks import *
import scipy.io
import os
import h5py
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

cqcc_feature = "..."

model = load_model('xxxxxxxxxx.h5')
bottleneck_model = Model(inputs=model.input, outputs=model.get_layer('dense_3').output)
bottleneck_features = bottleneck_model.predict(cqcc_feature, batch_size=1, verbose=1)
bottleneck_features = np.array(bottleneck_features)
scipy.io.savemat('bottleneck_features.mat', mdict={'bottleneck_features': bottleneck_features})


