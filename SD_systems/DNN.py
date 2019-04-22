import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import scipy.io
import os
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



x_train, y_train = "..."
x_test, y_test = "..."

model = Sequential()
model.add(Conv2D(128, (90, 3), activation='relu', input_shape=(90, 200, 1)))
model.add(Conv2D(128, (1, 3), activation='relu'))
model.add(Conv2D(128, (1, 3), activation='relu'))
model.add(Reshape((194,128)))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=100, epochs=50,validation_split=0.1)

score = model.predict(x_test, batch_size=1)
print(model.metrics_names)
print(score)

model.save('xxxxxxxxxx.h5')

