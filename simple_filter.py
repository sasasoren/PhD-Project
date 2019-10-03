from spyder_window_maker import create_featureset_label, int_to_mat
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
import matplotlib
import datetime
import os


win_size = 1

n_nodes_hl1 = 7







img = cv2.imread('org1-1.jpg', 0)

mat_contents = sio.loadmat('lab_mat_wo')
inside_m = mat_contents['labeled_mat']
class_mat = inside_m[49:250, 49:250]
print(class_mat[0,0])
img_cut = img[49:250, 49:250]

train_x, train_y, test_x, test_y, x_, y_  = create_featureset_label(img, img_cut, class_mat)
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


predict = neural_network_model(x_.astype('float'))
prob = tf.nn.softmax(neural_network_model(x_.astype('float')))

pred_idx = tf.argmax(predict, 1)

y_idx = tf.argmax(y_, 1)
correct = tf.equal(pred_idx , y_idx)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(predict, 1), num_classes=2)







with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())


    print('Accuracy:', accuracy.eval())

    print('Confusion:',confusion.eval())


    img_prob = prob[:,0].eval().reshape(img_cut.shape) *255
    img_bin = pred_idx.eval().reshape(img_cut.shape) * 255
    #
    # cv2.imwrite("/home/sorena/Research/report/Images/img_prob_sim.png", img_prob)
    # cv2.imwrite("/home/sorena/Research/report/Images/img_bin_sim.png", img_bin)


img_bin[img_bin == 0] = 100
img_bin[img_bin == 255] = 0
img_bin[img_bin == 100] = 255


img_bin = img_bin/255
l = img_bin.shape
for i1 in range(np.shape(img_bin)[0] - 2):
    for j1 in range(np.shape(img_bin)[0] - 2):
        i = i1 + 1
        j = j1 + 1
        if img_bin[i-1,j] == img_bin[i+1,j] == img_bin[i,j+1] == img_bin[i,j-1]:
            img_bin[i,j] = img_bin[i-1,j]
cor = np.zeros(l)
cor[img_bin == class_mat] = 1
acc = cor.sum()/(l[0]*l[1])



with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())

    print('Accuracy:', acc)
    img_bin = img_bin * 255

    cv2.imwrite("/home/sorena/Research/report/Images/img_bin_sim.png", img_bin)
