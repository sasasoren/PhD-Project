from spyder_window_maker import create_featureset_label, win_ftset_and_label, vote_filter
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
from MLP_after_auto import neural2
from auto_MLP import neural1
from sklearn.metrics import accuracy_score, confusion_matrix
import datetime
import os
# from auto_MLP import neural_network_model
from auto_MLP import nn_model
Conf=[]
Acc = []
Conf_fil=[]
Acc_fil = []
predic_image = []
win_size = 1
h_filter = 10
path = os.getcwd()
model_path = os.path.join(path, "model_save", "model.ckpt")
print(model_path)

tf.reset_default_graph()
# print("time = ", t, "image number = ", ij)
mat_contents = sio.loadmat('mum-perf-org-new-1-34-t=1-3.mat')

y = mat_contents['A'][0][0][0][0]
img = mat_contents['A'][0][2][0][0]

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
                 l1_eval_x.astype('float32'), hm_epochs= 100)

with tf.Session() as sess:
    global img_bin
    img_bin = sess.run(tf.argmax(x_pred, 1)).reshape(img_cut.shape) * 125
    print('img_bin.unique',np.unique(img_bin))
    cv2.imwrite("/home/sorena/Documents/Spyderthon/Project/Homayun/Images/img_bin_3class.png", img_bin)

conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

Acc.append(acc)
Conf.append(conf)
tf.reset_default_graph()

print('Accuracy for all images',Acc)

print('Confusion for all images',Conf)


d = img_bin.shape


train_x1, train_y1, test_x1, test_y1, x1_, y1_,l_x = win_ftset_and_label(img_bin, img_bin[win_size:d[0]-win_size,win_size:d[1]-win_size], class_mat)
print('label_x : ',l_x[0])

x1_pred, acc_fil, conf_fil = neural2(train_x1, train_y1, test_x1, test_y1, x1_, hm_epochs = 100 ,n_nodes_hl1 = h_filter)


with tf.Session() as sess:
    img_bin_fil = sess.run(tf.argmax(x1_pred, 1)).reshape(img_bin[win_size:d[0]-win_size,win_size:d[1]-win_size].shape) * 125

    cv2.imwrite("/home/sorena/Documents/Spyderthon/Project/Homayun/Images/img_bin_fil_3class.png", img_bin_fil)


conf_fil = conf_fil.astype('float') / conf_fil.sum(axis=1)[:, np.newaxis]
Acc.append(acc_fil)
Conf.append(conf_fil)
tf.reset_default_graph()


cut_img_bin = img_bin[win_size:d[0]-win_size,win_size:d[1]-win_size]
class_test = class_mat[win_size:d[0]-win_size,win_size:d[1]-win_size]

# fil_img = np.zeros(np.shape(cut_img_bin))
#
# for i, j in np.ndindex(np.shape(cut_img_bin)):
#     i1 = i + win_size
#     j1 = j + win_size
#
#     A = np.ndarray.flatten(img_bin[i1 - win_size:i1 + win_size + 1, j1 - win_size:j1 + win_size + 1])
#
#     fil_img[i,j] = np.argmax(np.bincount(A))
#
img_fil = img_bin/125
fil_img = vote_filter(img_fil, thrshold= 7)

print('fil_img.shape inside the vote function',np.shape(fil_img))



print(img_fil[0])

cv2.imwrite("/home/sorena/Documents/Spyderthon/Project/Homayun/Images/img_bin_cut_votefil.png", fil_img)
fil_img[fil_img == 0] = 3
fil_img[fil_img == 1] = 0
fil_img[fil_img == 3] = 1


cm = confusion_matrix(np.ndarray.flatten(class_mat),np.ndarray.flatten(fil_img))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('unique class_test:', np.unique(np.ndarray.flatten(class_mat)))
print('unique fil_img:', np.unique(np.ndarray.flatten(fil_img)))

ac = accuracy_score(np.ndarray.flatten(class_mat),np.ndarray.flatten(fil_img))


print('confucion matrix value:' , cm)
Acc.append(ac)
Conf.append(cm)



print('Accuracy for all images',Acc)

print('Confusion for all images',Conf)
