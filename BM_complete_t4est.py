#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:18:14 2018

@author: sorena
"""
#import matplotlib
#matplotlib.use('agg')
import cv2
import numpy as np
#from spyder_window_maker import data_iterator
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.io as sio

bm_epoch = 300
test_num = 5
num_classes = 2
start_point = 49
T = 10
save_dir = "Images/BM_test/"


img_num = [0, 7, 18, 33, 43, 58]

img_idx = cv2.imread('x_idx.png', 0)/255
img_org = cv2.imread('x_pr.png', 0)/255

sec_prb = cv2.imread('x_pr_sec.png', 0)/255
sec_idx = cv2.imread('CNN58.png', 0)/255

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

l = np.shape(sec_idx)

pix_pos = [(i, j) for i, j in np.ndindex((l[0]-2, l[1]-2))]
pix = np.add(np.array(pix_pos), 1)
    
w = np.load('w.npy')
v = np.load('v.npy')
v_prb = np.load('v_prb.npy')
bv = np.load('bv.npy')
bh = np.load('bh.npy')


vis = sec_idx.copy()
hid = sec_idx.copy()
#hid = np.zeros(np.shape(vis))
out_vis = np.zeros(np.shape(vis))


def new_node_hid(pixel, visible, img_prb, W, V_prb, bh, t):
    vis_neighbor = visible[pixel[0]-1:pixel[0]+2, pixel[1]-1:pixel[1]+2]

    prb_neighbor = img_prb[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]

    delta_E = np.sum(np.multiply(vis_neighbor, W)) +\
              np.sum(np.multiply(prb_neighbor, V_prb)) + bh

    prb = 1 / (1 + np.exp(-(delta_E)/t))
    s_new = np.random.choice((0, 1), 1, p=[1 - prb, prb])
    return s_new


def new_node_vis(pixel, visible, hidden, W, V, bv, t):
    vis_neighbor = visible[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    vis_neighbor[1, 1] = 0
    hid_neighbor = hidden[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    prb = 1 / (1 + np.exp(-(np.sum(np.multiply(vis_neighbor, W)) +
                            np.sum(np.multiply(hid_neighbor, V)) + bv)/t))
    s_new = np.random.choice((0, 1), 1, p=[1 - prb, prb])
    return s_new

ac = accuracy_score(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(sec_idx))
print('accuracy: ', ac)

for ep in tqdm(range(bm_epoch)):
    T = 1/(1.3 ** bm_epoch)

    s_hid1 = np.array(list(map(lambda x: new_node_hid(
                        x, vis, sec_prb, w, v_prb, bh, T), pix)))

    hid[pix[:, 0], pix[:, 1]] = s_hid1.reshape(np.shape(
            hid[pix[:, 0], pix[:, 1]]))

    s_vis1 = np.array(list(map(lambda x: new_node_vis(x, vis, hid, w,
                                                      v, bv, T), pix)))

    vis[pix[:, 0], pix[:, 1]] = s_vis1.reshape(
        np.shape(vis[pix[:, 0], pix[:, 1]]))
    ac = accuracy_score(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(vis))
    print('accuracy: ', ac )
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(vis)

    new_vis = vis.copy()
    new_vis[class_mat2_sec == 2] = 2
    plt.subplot(122)
    plt.imshow(new_vis)
    plt.savefig(save_dir + "test_"+str(ep)+".png")
    plt.show()
    if ep > 200:
        out_vis += vis
        

out_vis = out_vis / (bm_epoch - 201)
out_vis[out_vis > .5] = 1
out_vis[out_vis <= .5] = 0   


print('INPUT')
cm = confusion_matrix(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(sec_idx))
print('confucion matrix value:', cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('confucion matrix value:', cm)


ac = accuracy_score(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(sec_idx))
print('accuracy: ', ac )



print('OUT_VIS')
cm = confusion_matrix(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(out_vis))
print('confucion matrix value:', cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('confucion matrix value:', cm)


ac = accuracy_score(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(out_vis))
print('accuracy: ', ac )

#
#cv2.imwrite("Images/CNN" + str(img_num[test_num]) + ".png", img_bin_sec*255)
#image_show_border("Images/CNN" + str(img_num[test_num]) + ".png", class_mat_sec,
#                  "Images/mumCNN" + str(img_num[test_num]) + ".png")
