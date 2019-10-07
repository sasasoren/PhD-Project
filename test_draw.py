#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:39:57 2018

@author: sorena
"""

import cv2
import numpy as np
#from spyder_window_maker import data_iterator
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.io as sio


mat_contents = sio.loadmat('mum-perf-org-new1-600.mat')

y = mat_contents['B'][0][0]
img = mat_contents['B'][0][2]

cv2.imwrite('img-1-600.png', img*255)
img2 = cv2.imread('img-1-600.png')

cv2.rectangle(img2, (149,149), (449, 449), (255,0,0), 3)
cv2.imwrite('img-100-500.png', img2[100:500,100:500])
mat_contents = sio.loadmat('mum-perf-org-new150-300.mat')

y = mat_contents['B'][0][0]
img = mat_contents['B'][0][2]
cv2.imwrite('img-150-300.png', img*255)