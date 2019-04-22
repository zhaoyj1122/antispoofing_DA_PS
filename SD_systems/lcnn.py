
import numpy as np
import keras
from keras.layers import Input, Dense, Permute, Maximum, Reshape, Add, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Cropping2D, Cropping1D
from keras.models import Model, load_model
from keras.callbacks import *
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import scipy.io
import os
import h5py
import cPickle as pickle
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

timestr = time.strftime("%Y%m%d-%H%M%S")


# GPU # 
GPU = "1"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

batch_size = 128
num_epochs = 100
do_ratio = .5
lr_rate = 1e-4

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory()
validation_generator = test_datagen.flow_from_directory()

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer='glorot_normal', name=name)(x)
  if not use_bias:
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  return x

def mfm(x):
  shape = K.int_shape(x)
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x1 = Cropping2D(cropping=((0, shape[3] // 2), 0))(x)
  x2 = Cropping2D(cropping=((shape[3] // 2, 0), 0))(x)
  x = Maximum()([x1, x2])
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x = Reshape([shape[1], shape[2], shape[3] // 2])(x)
  return x

def common_conv2d(net, filters, filters2, iter=1):
  res = net

  for v in range(iter):
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = Add()([net, res]) # residual connection

  net = conv2d_bn(net, filters=filters, kernel_size=1, strides=1, padding='same')
  net = mfm(net)
  net = conv2d_bn(net, filters=filters2, kernel_size=3, strides=1, padding='same')
  net = mfm(net)

  return net

def lcnn29(inputs):
  # Conv1
  net = conv2d_bn(inputs, filters=32, kernel_size=5, strides=1, padding='same')
  net = mfm(net)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block1
  net = common_conv2d(net,filters=32, filters2=48, iter=0)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block2
  net = common_conv2d(net,filters=48, filters2=64, iter=0)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block3
  net = common_conv2d(net,filters=64, filters2=32, iter=0)

  # Block4
  net = common_conv2d(net,filters=32, filters2=32, iter=0)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)
  
  net = Flatten()(net)

  return net

input_image = Input(shape=(864, 400, 1))

lcnn_output = lcnn29(inputs=input_image)

fc1 = Dropout(do_ratio)(lcnn_output)            # Dropout layers
fc1 = Dense(64, activation=None)(fc1)
fc1 = Dropout(do_ratio)(fc1)                    # Dropout layers
fc1 = Reshape((64, 1))(fc1)
fc1_1 = Cropping1D(cropping=(0, 32))(fc1)
fc1_2 = Cropping1D(cropping=(32, 0))(fc1)
fc1 = Maximum()([fc1_1, fc1_2])
fc1 = Flatten()(fc1)
out = Dense(1, activation='sigmoid')(fc1)

model = Model(inputs=[input_image], outputs=out)
# model.summary()


Adam = keras.optimizers.Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam,
              # loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])


'''
# checkpoint
filepath = "model_" + timestr + "_fit_generator" + ".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
'''

history_callback = model.fit_generator(
        train_generator,
        steps_per_epoch=10000 // batch_size,
        epochs=num_epochs,
        #callbacks=callbacks_list,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=1510 // batch_size)



loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
# np.savetxt("loss_history_" + timestr + "_fit_generator" + ".txt", numpy_loss_history, delimiter=",")























