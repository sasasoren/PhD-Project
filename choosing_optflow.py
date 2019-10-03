import cv2
import numpy as np
from conv_func import conv_fun, label_img,\
    col_cell, draw_flow
import scipy.io as sio
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import sys
import compare_optical_flow as comp
from scipy.interpolate import griddata
from latex import insert_img


addad = 73

mat_contents = sio.loadmat('mum-perf-org-new-1-119.mat')

# y = mat_contents['A'][0][0][0][0]
# img2 = mat_contents['A'][0][2][0][0]
#
# y_sec = mat_contents['A'][0][0][1][0]
# img2_sec = mat_contents['A'][0][2][1][0]


save_add = "Images/compare_opt_flow/0/"


x_idx_sec = cv2.imread(save_add + "x_idx_sec.png", 0)

x_idx = cv2.imread(save_add + "x_idx.png", 0)


img = cv2.imread(save_add + "optic1.png", 1)
l = np.shape(img)
x_idx[0, :] = 0
x_idx[:, 0] = 0
x_idx[l[0]-1, :] = 0
x_idx[:, l[1]-1] = 0

img_sec = cv2.imread(save_add + "optic2.png", 1)

img2 = cv2.imread(save_add + "optic1.png", 0)

img_sec2 = cv2.imread(save_add + "optic2.png", 0)

# idx_label = label(x_idx, connectivity=1)
# x, y = np.where(idx_label == 70)
# obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
# flow_DIS = obj.calc(img2, img_sec2, None)
# fx, fy = flow_DIS[y, x].T
save_add = "Images/repo/"

img_pyr2 = cv2.pyrDown(img2)
img_pyr_sec2 = cv2.pyrDown(img_sec2)
img_pyr, img_pyr_sec, new_label, pr_mat = comp.down_labeled_img(img, img_sec, x_idx)
img_pyr_dob = cv2.pyrDown(img_pyr)
img_pyr_sec_dob = cv2.pyrDown(img_pyr_sec)

obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
flow_DIS = obj.calc(img2, img_sec2, None)
flow_DIS_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

obj = cv2.optflow.DualTVL1OpticalFlow_create()
flow_Dual = obj.calc(img2, img_sec2, None)
flow_Dual_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

obj = cv2.optflow.createOptFlow_DeepFlow()
flow_Deep = obj.calc(img2, img_sec2, None)
flow_Deep_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

obj = cv2.optflow.createOptFlow_PCAFlow()
flow_PCA = obj.calc(img2, img_sec2, None)
flow_PCA_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

obj = cv2.optflow.createOptFlow_SimpleFlow()
flow_SF = obj.calc(img2, img_sec2, None)
flow_SF_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

flow_Farne = cv2.calcOpticalFlowFarneback(img2, img_sec2, None,
                                          0.5, 3, 15, 3, 5,
                                          1.2, 0)


flow_Farne_pyr = cv2.calcOpticalFlowFarneback(img_pyr2, img_pyr_sec2, None,
                                              0.5, 3, 15, 3, 5,
                                              1.2, 0)

test_img = img.copy()
test_img_pyr = img_pyr.copy()
test_img_pyr_sec = img_pyr_sec.copy()
test_img_sec = img_sec.copy()

flow = [flow_Farne, flow_DIS, flow_Dual, flow_Deep, flow_PCA] #flow_SF,
        # flow_Farne_pyr, flow_DIS_pyr, flow_Dual_pyr, flow_Deep_pyr,
        # flow_PCA_pyr]#, flow_SF_pyr]
flow_name = ["Farne", "DIS", "DualTV", "Deepflow", "PCA"]
             # "Farne_pyr", "DIS_pyr", "DualTV_pyr", "Deepflow_pyr",
             # "PCA_pyr"]

idx_label = label(x_idx, connectivity=1)
cell_smooth = np.zeros((5, np.max(idx_label)))
for j in range(5):
    for i in range(np.max(idx_label)):
        cell_smooth[j, i] = comp.cell_smoothness(flow[j], x_idx,
                                              name_of_flow="DIS",
                                              cell_num=i,
                                              save_name="DIS",
                                              address=save_add,
                                              whole_=False)

    plt.hist(cell_smooth[j, :])
    plt.savefig("Images/diff_flow/hist" + str(flow_name[j]) + ".png")
    plt.show()
    new_add = "{diff_flow/hist" + flow_name[j] + "}"
    caption = "Histogram of smoothness of cells by " + flow_name[j] + " method. "
    insert_img(.8, new_add, caption)





# print("Name Of Flow & Gradient & Smoothness")
# for i in range(len(flow)):
#     if i == 5:
#         new_label[new_label != 0] = 1
#         x_idx = new_label
#         img2 = img_pyr2
#         img_sec2 = img_pyr_sec2
#     grad_value = comp.grad_intense(img2, img_sec2, flow[i], x_idx,
#                                    save_name=str(flow_name[i]),
#                                    name_of_flow=str(flow_name[i]),
#                                    white_=True, address=save_add, whole=1)
#
#     smoothness = comp.cell_smoothness(flow[i], x_idx,
#                                       name_of_flow=str(flow_name[i]),
#                                       cell_num=addad,
#                                       save_name=str(flow_name[i]),
#                                       address=save_add,
#                                       whole_=True)
#
#     print(str(flow_name[i]), "& {:.2}".format(grad_value), "&", "{:.2}".format(smoothness),"\\\\")
#     print("\\hline")

# for i in range(len(flow)):
#
#     if i > 4:
#         new_label[new_label != 0] = 1
#         x_idx = new_label
#         img2 = img_pyr2.copy()
#         img_sec2 = img_pyr_sec2.copy()
#     save_address = "Images/diff_flow/" + flow_name[i] + ".png"
#     comp.interpolate_img(img2, flow[i],
#                          save_address=save_address,
#                          method='linear')
#     new_add = "{diff_flow/cell40" + flow_name[i] + "}"
#     caption = "Interpolated of image with " + flow_name[i] + "flow"
#     insert_img(.8, new_add, caption)
# save_address = "Images/diff_flow/"
# for i in range(len(flow)):
#     test_img = img.copy()
#     test_img_sec = img_sec.copy()
#     if i > 4:
#         new_label[new_label != 0] = 1
#         x_idx = new_label
#         test_img = img_pyr.copy()
#         test_img_sec = img_pyr_sec.copy()
#
#     comp.grad_flow(flow[i], name_of_flow="1",
#               save_name="smoth_"+flow_name[i], address=save_address)
#     new_add = "{diff_flow/" + "smoth_"+flow_name[i] + "}"
#     caption = "Image for the smoothness of velocity by $" + flow_name[i] + "$ method. "
#     insert_img(.8, new_add, caption)
#     new_add = "{diff_flow/" + "smoth_" + flow_name[i] + "x}"
#     caption = "Image for the smoothness of x-axis velocity by $" + flow_name[i] + "$ method. "
#     insert_img(.8, new_add, caption)
#     new_add = "{diff_flow/" + "smoth_" + flow_name[i] + "y}"
#     caption = "Image for the smoothness of y-axis velocity by $" + flow_name[i] + "$ method. "
#     insert_img(.8, new_add, caption)



# par_ = [.1, .3, .5, .8, 1, 0, 10, 5]
# par_ = [0, 10, 100, 1000]
# par_ = [cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
#         cv2.DISOPTICAL_FLOW_PRESET_FAST,
#         cv2.DISOPTICAL_FLOW_PRESET_MEDIUM]
# print("\\hline")
# print("\\text{value of \\alpha} & Gradient & Smoothness")
# print("\\hline")
# for p in par_:
#     obj = cv2.DISOpticalFlow.create(p)
#     flow_DIS = obj.calc(img2, img_sec2, None)
#     # flow_DIS_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)
#     grad_value = comp.grad_intense(img2, img_sec2, flow_DIS, x_idx,
#                                    save_name="flow_DIS",
#                                    name_of_flow="flow_DIS",
#                                    white_=True, address=save_add, whole=1)
#
#     smoothness = comp.cell_smoothness(flow_DIS, x_idx,
#                                       name_of_flow="flow_DIS",
#                                       cell_num=addad,
#                                       save_name="flow_DIS",
#                                       address=save_add,
#                                       whole_=True)
#
#     print(p, "& {:.2}".format(grad_value), "&", "{:.2}".format(smoothness), "\\\\")
#     print("\\hline")
#
#










# comp.grad_flow(flow_DIS, name_of_flow="DIS",
#                save_name="DIS", address=save_add)




# x, y = np.where(x_idx == x_idx)
# grid_x, grid_y = np.mgrid[0:l[0], 0:l[1]]
# fx, fy = flow_DIS[y, x].T
# lines = np.vstack([x, y, x + fx, y + fy]) \
#     .T.reshape(-1, 2, 2)
# lines = np.int32(lines + 0.5)
# lines = list(filter(lambda w: np.all(w < l[0]), lines))
# lines = list(filter(lambda w: np.all(w >= 0), lines))
# lines = np.array(lines)
# pts = (lines[:, 1, 0], lines[:, 1, 1])
# prepts = (lines[:, 0, 0], lines[:, 0, 1])
# values = img2[prepts]
#
# img_interpolate = griddata(pts, values, (grid_x, grid_y), method='linear')
# plt.imshow(img_interpolate)
# plt.show()





















