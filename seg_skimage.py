import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import mahotas as mh
import cv2 as cv2

# img = cv2.imread("pic/1.tif", -1)
# img = cv2.normalize(img, dst=None, alpha=0, beta=65535,
#                     norm_type=cv2.NORM_MINMAX)

# img = cv2.imread("pic/31.tif", -1)


def main(img, size):
    img = img[size[0]:size[1], size[2]:size[3]]
    img = cv2.normalize(img, dst=None, alpha=0, beta=65535,
                        norm_type=cv2.NORM_MINMAX)

    #finding thresh hold with looking at .ravel() and trail and error
    ret, thresh = cv2.threshold(img, 7000, 65535, cv2.THRESH_BINARY_INV)


    # Using morphology for removing small noises
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 5)
    # # distance = ndi.distance_transform_edt(opening)
    # local_maxi = peak_local_max(dist_transform, indices=False, footprint=np.ones((3, 3)),
    #                             labels=img)
    # markers = ndi.label(local_maxi)[0]
    label_objects, nb_labels = ndi.label(opening)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 50
    mask_sizes[0] = 0
    new_open = mask_sizes[label_objects]
    nlabel_objects, nnb_labels = ndi.label(new_open)

    segmentation = watershed(-dist_transform, nlabel_objects, mask=sure_bg)

    # plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    # for i in range(1, np.max(segmentation)+1):
    #     plt.contour(segmentation == i, [0.5], linewidths=1.2, colors='y')
    # plt.axis('off')
    # plt.show()

    # new_seg = segmentation.copy()
    # for i in range(np.max(segmentation)+1):
    #     if np.min(np.where(segmentation == i)) == 0:
    #         new_seg[segmentation == i] = 0
    #     elif np.min(np.where(segmentation == i)) == np.max(np.shape(img)):
    #         new_seg[segmentation == i] = 0

    new_seg = mh.labeled.remove_bordering(segmentation)
    relabeled, n_left = mh.labeled.relabel(new_seg)

    plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    for i in range(1, np.max(relabeled)+1):
        plt.contour(relabeled == i, [0.5], linewidths=1.2, colors='y')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    imag = cv2.imread("pic/qual/4-2-19-c335yfp_c163_ribosexy1c1t001.tif", -1)
    main(imag, [700, 1000, 250, 750])
    # for i in range(10):
    #     imag = cv2.imread("pic/3{}.tif".format(i), -1)
    #     size = [350, 750, 350, 750]
    #     main(imag, size)


