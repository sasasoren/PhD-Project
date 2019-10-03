from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from MLP_after_auto import neural2
from auto_MLP import neural1
import datetime
import os
# from auto_MLP import neural_network_model
from auto_MLP import nn_model


Conf=[]
Acc = []
predic_image = []
path = os.getcwd()
model_path = os.path.join(path, "model_save", "model.ckpt")
print(model_path)
for t in range(1):
    for ij in range(1):
        tf.reset_default_graph()
        print("time = ", t, "image number = ", ij)
        mat_contents = sio.loadmat('mum-perf-org-new-1-34-t=1-3.mat')

        y = mat_contents['A'][0][0][t][ij]
        img = mat_contents['A'][0][2][t][ij]

        y[y==4] = 1
        class_mat = y[49:250 , 49:250]

        img_cut = img[49:250 , 49:250]

        train_x, train_y, test_x, test_y,x_,y_ = create_featureset_label(img, img_cut, class_mat)

        l1_eval_train, train_y, l1_eval_test, test_y, l1_eval_x, y_ = neural1(train_x, train_y, test_x, test_y, x_, y_)

        tf.reset_default_graph()



        x_pred, acc, conf = neural2(l1_eval_train.astype('float32'),
                         train_y.astype('float32'),
                         l1_eval_test.astype('float32'),
                         test_y.astype('float32'),
                         l1_eval_x.astype('float32'))

        with tf.Session() as sess:

            img_bin = sess.run(tf.argmax(x_pred, 1)).reshape(img_cut.shape) * 125

            cv2.imwrite("/home/sorena/Research/report/Images/img_bin_3class.png", img_bin)



        Acc.append(acc)
        Conf.append(conf)
        tf.reset_default_graph()

print('Accuracy for all images',Acc)

print('Confusion for all images',Conf)


