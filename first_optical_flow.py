import cv2
import numpy as np
from conv_func import conv_fun, label_img,\
    col_cell, draw_flow
import scipy.io as sio
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


cell_num = 0

save_add2 = "Images/first_optical_flow/"
save_add = "Images/first_optical_flow/" + str(cell_num)

mat_contents = sio.loadmat('mum-perf-org-new-1-119.mat')
win_num = 0
y = mat_contents['A'][0][0][0][win_num]
img2 = mat_contents['A'][0][2][0][win_num]

y_sec = mat_contents['A'][0][0][1][win_num]
img2_sec = mat_contents['A'][0][2][1][win_num]


# x_idx, x_idx_sec, img3, img3_sec \
#     = conv_fun(img2, y, img2_sec, y_sec,
#                img_width=201,
#                img_length=201,
#                save_add=save_add2,
#                pr_num_fil=3)
#
#
# cv2.imwrite(save_add2 + "x_idx_sec.png", x_idx_sec * 255)
# cv2.imwrite(save_add2 + "x_idx.png", x_idx * 255)
# cv2.imwrite(save_add2 + "optic1.png", img3 * 255)
# cv2.imwrite(save_add2 + "optic2.png", img3_sec * 255)
for cell_num in range(30, 85, 5):

    x_idx_sec = cv2.imread(save_add2 + "x_idx_sec.png", 0)

    x_idx = cv2.imread(save_add2 + "x_idx.png", 0)

    img = cv2.imread(save_add2 + "optic1.png", 1)

    img_sec = cv2.imread(save_add2 + "optic2.png", 1)

    img2 = cv2.imread(save_add2 + "optic1.png", 0)

    img_sec2 = cv2.imread(save_add2 + "optic2.png", 0)

    # img_sec2 = cv2.pyrDown(img_sec2)
    # img = cv2.pyrDown(img)
    # img_sec = cv2.pyrDown(img_sec)
    # img2 = cv2.pyrDown(img2)
    # x_idx = cv2.pyrDown(x_idx)
    # x_idx_sec = cv2.pyrDown(x_idx_sec)

    # x_idx[x_idx < 126] = 0
    # x_idx[x_idx > 1] = 1
    #
    # x_idx_sec[x_idx_sec < 126] = 0
    # x_idx_sec[x_idx_sec > 1] = 1

    labeled_img = label(x_idx, connectivity=1)
    regions = regionprops(labeled_img)
    labeled_img_sec = label(x_idx_sec, connectivity=1)
    regions_sec = regionprops(labeled_img_sec)

    # Lucas kanade params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # def mf_show(cell_num, img, img_sec):
    # x = regions[cell_num].centroid
    x = [0, 0]
    Z = np.where(labeled_img == cell_num)
    x[1] = int(np.sum(Z[0]) / len(Z[0]))
    x[0] = int(np.sum(Z[1]) / len(Z[1]))

    old_points = np.array([[x[0], x[1]]], dtype=np.float32)

    new_point, status, error = cv2.calcOpticalFlowPyrLK\
        (img, img_sec, old_points, None, **lk_params)

    obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow2 = obj.calc(img2, img_sec2, None)
    flow3 = cv2.calcOpticalFlowFarneback(img2, img_sec2, None,
                                         0.5, 3, 15, 3, 5,
                                         1.2, 0)
    x2, y2 = new_point.ravel()

    cv2.arrowedLine(img_sec, (int(x[0]), int(x[1])),
                    (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.circle(img_sec, (int(x[0]), int(x[1])), 1, (0, 0, 255), 1)
    test_img, _ = draw_flow(img2, flow2, point_=(x[1], x[0]))
    cv2.arrowedLine(test_img, (int(x[0]), int(x[1])),
                    (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.circle(img, (int(x[0]), int(x[1])), 2, (0, 0, 255), 2)

    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img)

    plt.subplot(212)
    plt.imshow(test_img)

    plt.savefig(save_add2 + str(cell_num) + "lk_img.png")
    plt.show()

    num_cell = labeled_img[x[1], x[0]]
    num_cell_sec = labeled_img_sec[int(y2), int(x2)]
    if num_cell == 0:
        print("Pixel is outside of the cell")
    else:
        col_img = img.copy()
        col_img[labeled_img == num_cell] = (0, 255, 255)
        plt.imshow(col_img)
        plt.savefig(save_add2 + str(cell_num) + "first_colored.png")
        plt.show()

        if num_cell_sec == 0:
            print("Pixel is outside of the cell")
        else:
            col_img_sec = img_sec.copy()
            col_img_sec[labeled_img_sec == num_cell_sec] = (0, 255, 255)
            plt.imshow(col_img_sec)
            plt.savefig(save_add2 + str(cell_num) + "second_colored.png")
            plt.show()

plt.imshow(draw_flow(img2, flow2))
plt.savefig(save_add2 + "_DIS_opt_flow.png")

plt.imshow(draw_flow(img2, flow3))
plt.savefig(save_add2 + "_Farne_opt_flow.png")




# cv2.imwrite(save_add + "img_sec.png", img_sec*255)
# cv2.imwrite(save_add + "img.png", img*255)
#
# new_img = cv2.imread(save_add + "img.png",0)/255
# min_img = np.minimum(new_img, x_idx)
# cv2.imwrite(save_add + "min_img.png", min_img * 255)

# mf_show(70, img, img_sec)

