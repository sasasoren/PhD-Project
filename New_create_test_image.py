from spyder_window_maker import create_featureset_label, int_to_mat
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
import matplotlib
import datetime
import os



img = cv2.imread('org600.png',0)

mat_contents = sio.loadmat('lab_mat_wo_600')
y=mat_contents['labeled_mat']
y[y==4] = 1
class_mat = y[49:550 , 49:550]

img_cut = img[49:550 , 49:550]

train_x, train_y, test_x, test_y,x_,y_  = create_featureset_label(img, img_cut, class_mat)
print(train_x.shape)



b1 = np.load('a_b1.npy').astype('float')
w1 = np.load('a_w1.npy').astype('float')
b2 = np.load('b1.npy').astype('float')
w2 = np.load('w1.npy').astype('float')
b3 = np.load('b2.npy').astype('float')
w3 = np.load('w2.npy').astype('float')

hidden_1_layer = {'weight': tf.Variable(w1,name= 'w1'),
                  'bias': tf.Variable(b1,name= 'b1')}
#print(hidden_1_layer['weight'])
hidden_2_layer = {'weight': tf.Variable(w2,name= 'w1'),
                  'bias': tf.Variable(b2,name= 'b1')}

output_layer = {'weight': tf.Variable(w3,name='w2'),
                'bias': tf.Variable(b3,name='b2') }


def neural_network_model(data):
    with tf.name_scope('Hidden_1'):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)

    with tf.name_scope('Hidden_2'):
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

    with tf.name_scope('Output'):
        output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output






prediction = neural_network_model(x_.astype('float'))

pred_idx = tf.argmax(prediction, 1)
y_idx = tf.argmax(y_, 1)
correct = tf.equal(pred_idx , y_idx)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(prediction, 1), num_classes=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    print('Accuracy:', accuracy.eval())

    print('Confusion:',confusion.eval())

    img_bin = pred_idx.eval().reshape(img_cut.shape) * 255

    cv2.imwrite("/home/sorena/Research/report/Images/img_bin600.png", img_bin)