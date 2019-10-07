from __future__ import print_function
from spyder_window_maker import win_ftset_and_label
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import os
import time



batch_size = 128
num_classes = 3
epochs = 50
start_point = 49
window_size = 3
# input image dimensions
img_rows, img_cols = 2*window_size+1, 2*window_size+1

img_num = [0, 7, 18, 33, 43, 58]

mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

y = mat_contents['B'][0][0][0][0]
img = mat_contents['B'][0][2][0][0]

y[y == 4] = 1
class_mat = y[start_point:start_point+201, start_point:start_point+201]

img_cut = img[start_point:start_point+201, start_point:start_point+201]

x_train, y_train, x_test, y_test, x_, y_, label_x = win_ftset_and_label(img, img_cut, class_mat
                                                             , test_size=.1, win_size=window_size
                                                             , win_tyoe='window_square'
                                                             , class_num=num_classes)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape[0], 'train samples')
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])