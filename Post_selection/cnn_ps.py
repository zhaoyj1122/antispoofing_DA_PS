import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import scipy.io
import os
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# GPU # 
GPU = "1"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
#set_session(tf.Session(config=config))

# ##########################################################################
x_train # real samples from T/D subsets + fake samples from DA generation pool
y_train # correspondind labels of x_train
x_test  # merged subsets for selection 
# ##########################################################################


model = Sequential()
model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(864, 400, 1)))
model.add(Conv2D(8, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=500, epochs=50,
          validation_split=0.1
          )

# model.save('model_CNN_selection.h5')

score = model.predict(x_test, batch_size=1)
print(model.metrics_names)
print(score)



