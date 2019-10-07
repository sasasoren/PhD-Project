from spyder_window_maker import win_ftset_and_label, image_show_border, vote_filter, BM_scratch, BM_predict
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 2
epochs = 20
start_point = 49
window_size = 3
learning_rate = 0.001
mon_freq = 100
test_num = 5
# input image dimensions
img_rows, img_cols = 2 * window_size + 1, 2 * window_size + 1

img_num = [0, 7, 18, 33, 43, 58]


mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

y = mat_contents['B'][0][0][0][33]
img = mat_contents['B'][0][2][0][33]

y_sec = mat_contents['B'][0][0][0][img_num[test_num]]
img_sec = mat_contents['B'][0][2][0][img_num[test_num]]

y[y == 4] = 1
y_sec[y_sec == 4] = 1
class_mat = y[start_point:start_point + 201, start_point:start_point + 201]
class_mat2 = class_mat.copy()
class_mat[class_mat == 2] = 0
class_mat_sec = y_sec[start_point:start_point + 201, start_point:start_point + 201]
class_mat2_sec = class_mat_sec.copy()
class_mat_sec[class_mat_sec == 2] = 0

img_cut = img[start_point:start_point + 201, start_point:start_point + 201]

img_bin = cv2.imread('Images/x_idx.png',0)
img_bin = img_bin/255

W1, W2, W3, b1, vis0 = BM_scratch(img_bin, img_cut,  class_mat, class_mat2, bm_epoch=15)
