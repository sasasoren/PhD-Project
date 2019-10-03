from spyder_window_maker import create_featureset_label, vote_filter
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from MLP_after_auto import neural2, neural_network_model
from auto_MLP import neural1
from PIL import Image
import datetime
import os
from sklearn.metrics import accuracy_score, confusion_matrix
# from auto_MLP import neural_network_model
from auto_MLP import nn_model
tex = []
Acc= []
Confu = []
predic_image = []
path = os.getcwd()
model_path = os.path.join(path, "model_save", "model.ckpt")


tf.reset_default_graph()


img_num = [0, 7, 18, 33, 43, 58]

mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

y = mat_contents['B'][0][0][0][0]
img = mat_contents['B'][0][2][0][0]

y[y==4] = 1
class_mat = y[49:250 , 49:250]

img_cut = img[49:250 , 49:250]

train_x, train_y, test_x, test_y,x_,y_ = create_featureset_label(img, img_cut, class_mat, class_num= 3)

l1_eval_train, train_y, l1_eval_test, test_y, l1_eval_x, y_ = neural1(train_x, train_y, test_x, test_y, x_, y_)

tf.reset_default_graph()



x_pred, acc, conf = neural2(l1_eval_train.astype('float32'),
                 train_y.astype('float32'),
                 l1_eval_test.astype('float32'),
                 test_y.astype('float32'),
                 l1_eval_x.astype('float32'), n_classes= 3)

tf.reset_default_graph()
#
# train_x, train_y, test_x, test_y,x_,y1_ = create_featureset_label(img, img_cut, class_mat)
#
# l1_eval_train, train_y, l1_eval_test, test_y, l1_eval_x, _ = neural1(train_x, train_y, test_x, test_y, x_, y1_)

x_whole = []
y_ = []
init = tf.global_variables_initializer()
print(y_)
for ij in range(6):
    print("time = ", 0, "image number = ", img_num[ij])

    y = mat_contents['B'][0][0][0][img_num[ij]]
    img = mat_contents['B'][0][2][0][img_num[ij]]

    y[y == 4] = 1
    class_mat = y[49:250, 49:250]

    img_cut = img[49:250, 49:250]

    _, _, _, _, x_, y_class = create_featureset_label(img, img_cut, class_mat, class_num= 3)

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

        print(type(x_))
        x_w = sess.run(l1, feed_dict={x: x_})
        x_whole.append(x_w)
        y_.append(y_class)
        print(np.shape(x_whole))
        print(np.shape(y_))







tf.reset_default_graph()

#
# x_pred, acc, conf = neural2(l1_eval_train.astype('float32'),
#                          train_y.astype('float32'),
#                          l1_eval_test.astype('float32'),
#                          test_y.astype('float32'),
#                          l1_eval_x.astype('float32'))
#



for ij in range(6):
    text = "time = " + str(0) + "image number = " + str(img_num[ij]) + ':\n'
    print(text)
    Acc.append(text)
    Confu.append(text)

    x = tf.placeholder('float')
    y = tf.placeholder('float')

    x_in = x_whole[ij + 2 * 0]
    y_in = y_[ij + 2*0]
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

        confusion = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(x_p, 1))

        x_pr = sess.run(x_p, feed_dict={x: x_in})

        accu = accuracy.eval({x: x_in, y: y_in})
        print('Accuracy:', accu)
        Acc.append(accu)

        confu = sess.run(confusion, feed_dict={x: x_in, y: y_in})
        print('Confusion:', confu)
        confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
        Confu.append(confu)

        img_bin = sess.run(tf.argmax(x_pr, 1)).reshape(img_cut.shape) * 125

        cv2.imwrite("/home/sorena/Research/report_oct_19/Images/3c_img_bin_other_time" + str(0) + "_num_" + str(img_num[ij]) + ".png",
                    img_bin)

        img_fil = img_bin / 125
        fil_img, _counter = vote_filter(img_fil, thrshold=6)

        cv2.imwrite("/home/sorena/Research/report_oct_19/Images/3c_fil_img_bin_other_time" + str(0) + "_num_" + str(img_num[ij]) + ".png",
                    fil_img)





    y = mat_contents['B'][0][0][0][img_num[ij]]
    y[y == 4] = 1
    class_mat = y[49:250, 49:250]

    class_mat[class_mat == 0] = 3
    class_mat[class_mat == 1] = 0
    class_mat[class_mat == 3] = 1

    cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(fil_img))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('unique class_test:', np.unique(np.ndarray.flatten(class_mat)))
    print('unique fil_img:', np.unique(np.ndarray.flatten(fil_img)))

    ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(fil_img))


    text2 = 'After filter for image number ' + str(img_num[ij]) + ':'

    Acc.append(text2)
    Acc.append(_counter)
    Confu.append(text2)
    Acc.append(ac)
    Confu.append(cm)


    mum_img = cv2.imread("/home/sorena/Research/report_oct_19/Images/3c_img_bin_other_time" + str(0) + "_num_" + str(img_num[ij]) + ".png",1)
    mum_img[class_mat == 2] = [0, 0, 255]

    por = Image.fromarray(mum_img)
    Address = "/home/sorena/Research/report_oct_19/Images/3c_img_bin_other_mum_time" + str(0) + "_num_" + str(img_num[ij]) + ".png"
    por.save(Address)

    mum_img = cv2.imread("/home/sorena/Research/report_oct_19/Images/3c_fil_img_bin_other_time" + str(0) + "_num_"
                         + str(img_num[ij]) + ".png", 1)
    mum_img[class_mat == 2] = [0, 0, 255]

    por = Image.fromarray(mum_img)
    Address = "/home/sorena/Research/report_oct_19/Images/3c_fil_img_bin_other_mum_time" + str(0) + "_num_" \
              + str(img_num[ij]) + ".png"
    por.save(Address)


print('Accuracy: ',Acc)
print('Confusion Matrix: ' , Confu)














