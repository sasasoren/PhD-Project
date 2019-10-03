from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from MLP_after_auto import neural2, neural_network_model
from auto_MLP import neural1
from PIL import Image
import datetime
import os
# from auto_MLP import neural_network_model
from auto_MLP import nn_model

tex = []
Acc= []
Confu = []
x_whole = []
y_ = []
init = tf.global_variables_initializer()

for ij in range(2):
    tf.reset_default_graph()
    mat_contents = sio.loadmat('mum-perf-org-new-1-34-t=1-3.mat')

    y = mat_contents['A'][0][0][0][ij]
    img = mat_contents['A'][0][2][0][ij]

    y[y==4] = 1
    class_mat = y[49:250 , 49:250]

    img_cut = img[49:250 , 49:250]

    train_x, train_y, test_x, test_y,x_,y1_ = create_featureset_label(img, img_cut, class_mat, class_num= 2)
    print('test_y dim:', np.shape(test_y))
    l1_eval_train, train_y, l1_eval_test, test_y, l1_eval_x, _ = neural1(train_x, train_y, test_x, test_y, x_, y1_)

    for t in range(3):
        print("time = ", t, "image number = ", ij)

        y = mat_contents['A'][0][0][t][ij]
        img = mat_contents['A'][0][2][t][ij]

        y[y==4] = 1
        class_mat = y[49:250 , 49:250]

        img_cut = img[49:250 , 49:250]

        _, _, _, _, x_, y_class = create_featureset_label(img, img_cut, class_mat,class_num= 2)

        y_.append(y_class)

        x = tf.placeholder('float')

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph( 'auto_save/my_model.ckpt.meta')
            saver.restore(sess,  'auto_save/my_model.ckpt')
            # access a variable from the saved Graph, and so on:
            w1 = sess.run('a_w1:0')
            b1 = sess.run('a_b1:0')
            w2 = sess.run('a_w2:0')
            b2 = sess.run('a_b2:0')

            hidden_1_layer = {'weight':w1,'bias':b1}

            output_layer = {'weight':w2,'bias':b2}

            _, l1 = nn_model(x, hidden_1_layer, output_layer)

            x_w = sess.run(l1, feed_dict={x: x_})
            x_whole.append(x_w)









    tf.reset_default_graph()


    x_pred, acc, conf = neural2(l1_eval_train.astype('float32'),
                             train_y.astype('float32'),
                             l1_eval_test.astype('float32'),
                             test_y.astype('float32'),
                             l1_eval_x.astype('float32'), hm_epochs = 100, n_classes= 2)




    for t in range(3):
        text = "time = "+ str(t)+ "image number = "+ str(ij)
        print(text)
        Acc.append(text)
        Confu.append(text)

        x = tf.placeholder('float')
        y = tf.placeholder('float')

        x_in = x_whole[t + 3*ij]
        y_in = y_[t + 3*ij]
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph( 'mlp_save/my_model.ckpt.meta')
            saver.restore(sess,  'mlp_save/my_model.ckpt')
            # access a variable from the saved Graph, and so on:

            w1_ = sess.run('w1:0')

            b1_ = sess.run('b1:0')
            w2_ = sess.run('w2:0')
            b2_ = sess.run('b2:0')



            hidden_1_layer1 = {'weight':w1_,'bias':b1_}

            output_layer1 = {'weight':w2_,'bias':b2_}

            x_p = neural_network_model(x, hidden_1_layer1, output_layer1)
            pred_idx = tf.argmax(x_p, 1)
            y_idx = tf.argmax(y, 1)
            correct = tf.equal(pred_idx, y_idx)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            confusion = tf.confusion_matrix(labels=y_idx, predictions=pred_idx)

            pred_idx1 = sess.run(pred_idx, feed_dict={x: x_in})

            accu = accuracy.eval({x: x_in, y: y_in})
            print('Accuracy:', accu)
            Acc.append(accu)

            confu = sess.run(confusion, feed_dict={x: x_in, y: y_in})
            print('Confusion:', confu)
            confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
            Confu.append(confu)

            img_bin = pred_idx1.reshape(img_cut.shape) * 255

            cv2.imwrite("/home/sorena/Research/report/Images/img_bin_oneother2_time" + str(t) + "_num_" + str(ij) + ".png",
                        img_bin)

        y = mat_contents['A'][0][0][t][ij]
        y[y == 4] = 1
        class_mat = y[49:250, 49:250]

        mum_img = cv2.imread("/home/sorena/Research/report/Images/img_bin_oneother2_time" + str(t) + "_num_" + str(ij) + ".png",1)
        mum_img[class_mat == 2] = [0, 0, 255]

        por = Image.fromarray(mum_img)
        Address = "/home/sorena/Research/report/Images/img_bin_oneother2_mum_time" + str(t) + "_num_" + str(ij) + ".png"
        por.save(Address)







print(Acc)
print(Confu)