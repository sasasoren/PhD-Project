import scipy.io as sio
import numpy as np
import cv2


start_point = 49


mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

for i in range(60):
    y = mat_contents['B'][0][0][0][i]
    img = mat_contents['B'][0][2][0][i]

    y[y == 4] = 1

    class_mat = y[start_point:start_point + 201, start_point:start_point + 201]
    class_mat2 = class_mat.copy()
    class_mat[class_mat == 2] = 0

    img_cut = img[start_point:start_point + 201, start_point:start_point + 201]
    cv2.imwrite("image/"+str(i)+".png", img_cut * 255)
    cv2.imwrite("label/"+str(i)+".png", class_mat * 255)

