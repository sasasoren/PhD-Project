import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import watershed


# img = cv2.imread("pic/1.tif", -1)
# img = cv2.normalize(img, dst=None, alpha=0, beta=65535,
#                     norm_type=cv2.NORM_MINMAX)

img = cv2.imread("pic/30.tif", -1)
img = img[350:750, 350:750]
img = cv2.normalize(img, dst=None, alpha=0, beta=65535,
                    norm_type=cv2.NORM_MINMAX)
ret, thresh = cv2.threshold(img, 9000, 65535, cv2.THRESH_BINARY_INV)
thresh = (thresh / 257).astype(np.uint8)


# img2 = cv2.imread("pic/31.tif", -1)
# img2 = img2[350:750, 350:750]
# img2 = cv2.normalize(img2, dst=None, alpha=0, beta=65535,
#                     norm_type=cv2.NORM_MINMAX)
#
# plt.imshow(img)
# plt.show()
# plt.imshow(img2)
# plt.show()

# img = cv2.imread("pic/30.tif")
# img = img[350:750, 350:750]
# img = cv2.normalize(img, dst=None, alpha=0, beta=255,
#                     norm_type=cv2.NORM_MINMAX)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY_INV)


# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret1, sure_fg = cv2.threshold(dist_transform, 3.1, 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)


# Marker labelling
ret2, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# img = cv2.imread("pic/30.tif", 1)
# img = img[350:750, 350:750]
# img = cv2.normalize(img, dst=None, alpha=0, beta=255,
#                     norm_type=cv2.NORM_MINMAX)
labels = watershed(img, markers)
imgx = cv2.imread("pic/30.tif", 1)
imgx = imgx[350:750, 350:750]
imgx = cv2.normalize(imgx, dst=None, alpha=0, beta=255,
                     norm_type=cv2.NORM_MINMAX)
imgx[labels == -1] = [255, 0, 0]


plt.imshow(imgx)
plt.show()




