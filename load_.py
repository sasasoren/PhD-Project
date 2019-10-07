from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import datetime
import os
# from auto_MLP import neural_network_model
from auto_MLP import nn_model

path = os.getcwd()
model_path = os.path.join(path, "model_save", "model.ckpt")
print(model_path)
def read_and_save(pic1,class1):
    img = cv2.imread(pic1,0) #'org600.png'

    mat_contents = sio.loadmat(class1)#'lab_mat_wo_600'
    y=mat_contents['labeled_mat']
    y[y==4] = 1
    class_mat = y[49:550 , 49:550]

    img_cut = img[49:550 , 49:550]

    train_x, train_y, test_x, test_y,_,_ = create_featureset_label(img, img_cut, class_mat)
    x = tf.placeholder('float')

    # _, l3 = neural_network_model(x)
    _, l1 = nn_model(x)
    print(l1)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        load_path = saver.restore(sess, model_path)
        # print("Model restored from file: %s" % load_path)
        # print(load_path)
        l1_eval_train = sess.run(l1, feed_dict= {x: train_x})
        print(l1_eval_train.shape)
        np.save("Out_auto_x_train.npy",l1_eval_train)
        l1_eval_test = sess.run(l1, feed_dict={x: test_x})
        print(l1_eval_test.shape)
        np.save("Out_auto_x_test.npy", l1_eval_test)
        np.save("Out_auto_y_train.npy", train_y)
        np.save("Out_auto_y_test.npy", test_y)
