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
from latex import insert_img, insert_img2, insert_img3, img_table,\
    insert_img3_onecap, img8_input


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


img = cv2.imread(save_add + "optic1.png", 0)
l = np.shape(img)
x_idx[0, :] = 0
x_idx[:, 0] = 0
x_idx[l[0]-1, :] = 0
x_idx[:, l[1]-1] = 0

img_sec = cv2.imread(save_add + "optic2.png", 0)

# img2 = cv2.imread(save_add + "optic1.png", 0)
#
# img_sec2 = cv2.imread(save_add + "optic2.png", 0)


def for_show(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# idx_label = label(x_idx, connectivity=1)
# x, y = np.where(idx_label == 70)
# obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
# flow_DIS = obj.calc(img2, img_sec2, None)
# fx, fy = flow_DIS[y, x].T


# img_pyr2 = cv2.pyrDown(img2)
# img_pyr_sec2 = cv2.pyrDown(img_sec2)
# img_pyr, img_pyr_sec, new_label, pr_mat = comp.down_labeled_img(img, img_sec, x_idx)
# img_pyr_dob = cv2.pyrDown(img_pyr)
# img_pyr_sec_dob = cv2.pyrDown(img_pyr_sec)
#
# # obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# # flow_DIS = obj.calc(img2, img_sec2, None)
# # flow_DIS_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)
#
# flow_Farne = cv2.calcOpticalFlowFarneback(img2, img_sec2, None,
#                                           0.5, 3, 15, 3, 5,
#                                           1.2, 0)
#
#
# flow_Farne_pyr = cv2.calcOpticalFlowFarneback(img_pyr2, img_pyr_sec2, None,
#                                               0.5, 3, 15, 3, 5,
#                                               1.2, 0)

cell_list1 = [73, 67, 82, 90]
cell_list2 = [78, 66, 81, 94]
# cell_list1 = [39, 50]
# cell_list2 = [37, 45, 63]
#
# cell_list1 = [1]
# cell_list2 = [1]
#
# save_add = "Images/test_flow"

#
# lab_img = label(x_idx, connectivity=1)
# lab_img_sec = label(x_idx_sec, connectivity=1)
#
#
# lab_img[~np.in1d(lab_img.ravel(),
#                  cell_list1).reshape(lab_img.shape)] = 0
# lab_img_sec[~np.in1d(lab_img_sec.ravel(),
#                      cell_list2).reshape(lab_img_sec.shape)] = 0
#
# x_idx[lab_img == 0] = 0
# x_idx_sec[lab_img_sec == 0] = 0
#
#
# plt.imshow(x_idx)
# plt.show()
# plt.imshow(x_idx_sec)
# plt.show()
#
# obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# obj.setGradientDescentIterations(100)
# obj.setUseMeanNormalization(1)
# obj.setVariationalRefinementAlpha(10)
# flow_DIS = obj.calc(x_idx, x_idx_sec, None)
# x_idx = cv2.cvtColor(x_idx, cv2.COLOR_GRAY2BGR)
# x_idx_sec = cv2.cvtColor(x_idx_sec, cv2.COLOR_GRAY2BGR)
# for cell in cell_list1:
#     comp.trans_one_cel_lk(x_idx, x_idx_sec,
#                           new_seg,
#                           which_num=cell,
#                           save_name="test",
#                           lab_img=None,
#                           address=save_add)

# img = np.zeros((100, 100), np.uint8)
# cv2.ellipse(img, (50, 50), (10, 5), 0, 0, 360, 255, -1)
# img_idx = img.copy()
# img = cv2.GaussianBlur(img, (5, 5), 0)
#
# img_sec = np.zeros((100, 100), np.uint8)
# cv2.ellipse(img_sec, (50, 50), (14, 7), 0, 0, 360, 255, -1)
# img_sec_idx = img_sec.copy()
# img_sec = cv2.GaussianBlur(img_sec, (5, 5), 0)

# img2[10:15, 40:80] = 255
# img_sec2[20:25, 50:90] = 255
# img2[110:115, 140:180] = 255
# img_sec2[90:96, 120:185] = 255
#
save_add = "Images/test_flow/o"
new_add = "{test_flow/o"
scale1 = .5
scale2 = .3
img_show = for_show(img)
plt.imshow(img_show)
plt.savefig(save_add + "org.png")
plt.show()
img_sec_show = for_show(img_sec)
plt.imshow(img_sec_show)
plt.savefig(save_add + "org2.png")
plt.show()
# sys.stdout = open(address + "grade_intense", "a")
new_add1 = new_add + "org.png}"
new_add2 = new_add + "org2.png}"
caption1 = "Original Image at t=0. "
caption2 = "Original Image at t=1. "
insert_img2(scale1, new_add1, new_add2, caption1, caption2)

flow_name = ["Farneback", "DIS", "DualTV", "Deep", "PCA", "SF"]

for i in range(6):
    test_img = img.copy()
    test_img_sec = img_sec.copy()

    org1, org2, pred = comp.some_cell_flow(test_img, test_img_sec,
                                           x_idx, x_idx_sec,
                                           cell_list1, cell_list2,
                                           flow_num=i, lucas=None,
                                           save_address=save_add)
    img_adds = []
    # plt.imshow(org1)
    # plt.savefig(save_add + "org.png")
    # plt.imshow(org2)
    # plt.savefig(save_add + "org2.png")
    #
    diff_0 = np.uint8(255 - np.abs(org1 - org2))
    diff_0 = for_show(diff_0)
    plt.imshow(diff_0)
    plt.savefig(save_add + str(i) + "diff0.png")
    diff_1 = np.uint8(255 - np.abs(org1 - pred))
    diff_1 = for_show(diff_1)
    plt.imshow(diff_1)
    plt.savefig(save_add + str(i) + "diff1.png")

    diff_2 = np.uint8(255 - np.abs(org2 - pred))
    diff_2 = for_show(diff_2)
    plt.imshow(diff_2)
    plt.savefig(save_add + str(i) + "diff2.png")
    # we define address of images for latex file i img_adds
    img_adds.append(new_add + str(i) + "diff0.png}")
    img_adds.append(new_add + str(i) + "diff1.png}")
    img_adds.append(new_add + str(i) + "diff2.png}")
    img_adds.append(new_add + str(i) + "predict_sec.png}")
    img_adds.append(new_add + str(i) + "interpolate.png}")
    img_adds.append(new_add + str(i) + "flow.png}")
    img_adds.append(new_add + str(i) + "flowx.png}")
    img_adds.append(new_add + str(i) + "flowy.png}")
    # caption for latex
    caption = "The results for " + flow_name[i] + " flow. From top left to bottom right" \
                                                  " (1) difference of two original images" \
                                                  " (2) difference of prediction and image t=0" \
                                                  " (3) difference of prediction and image t=1" \
                                                  " (4) prediction image (5) interpolate image" \
                                                  " (6) l1 norm of flow (7) flow in x axis" \
                                                  " (8) flow in y axis"
    img8_input(0.3, img_adds, caption)

    #
    # new_add1 = new_add + str(i) + "predict_sec.png}"

    # new_add2 = new_add + str(i) + "interpolate.png}"
    # caption1 = "Predicted image with " + flow_name[i] + " by color blue on original image at time 1. "
    # caption2 = "Predicted image after interpolate of prediction. "
    # insert_img2(scale1, new_add1, new_add2, caption1, caption2)
    #
    # new_add1 = new_add + str(i) + "diff1.png}"
    # new_add2 = new_add + str(i) + "diff2.png}"
    # caption1 = "difference between Original image at time 0 and predicted" \
    #            " image with " + flow_name[i] + ". "
    # caption2 = "difference between Original image at time 1 and predicted" \
    #            " image with " + flow_name[i] + ". "
    # insert_img2(scale1, new_add1, new_add2, caption1, caption2)
    #
    # new_add1 = new_add + str(i) + "flowx.png}"
    # new_add2 = new_add + str(i) + "flowy.png}"
    # new_add3 = new_add + str(i) + "flow.png}"
    # caption1 = "x-axis vector field of " + flow_name[i] + " flow. "
    # caption2 = "y-axis vector field of " + flow_name[i] + " flow. "
    # caption3 = "Norm L2 vector field of " + flow_name[i] + " flow. "
    # insert_img3(scale2, new_add1, new_add2, new_add3,
    #             caption1, caption2, caption3)



# img_add=[]
# img_add.append(new_add + "org.png}")
# img_add.append(new_add + "org.png}")
# img_add.append(new_add + "org.png}")
# img_add.append(new_add + "org.png}")
# img_table(2, 2, img_add, "That's what it is")
# sys.stdout.close()















