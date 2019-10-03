
import cv2
import numpy as np
from spyder_window_maker import data_iterator
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.io as sio



#
# mat_contents = sio.loadmat('lab_mat_wo_600')
# y = mat_contents['labeled_mat']
# y[y == 4] = 1
# class_mat = y[49:550, 49:550]
# y[y == 2] = 0

img_idx = cv2.imread('x_idx.png', 0)/255
img_org = cv2.imread('x_pr.png', 0)/255



def new_node_hid(pixel, visible, img_prb, W, V_prb, bh):
    vis_neighbor = visible[pixel[0]-1:pixel[0]+2, pixel[1]-1:pixel[1]+2]

    prb_neighbor = img_prb[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]

    prb = 1 / (1 + np.exp(-(np.sum(np.multiply(vis_neighbor, W)) + np.sum(np.multiply(prb_neighbor, V_prb)) + bh)))
    s_new = np.random.choice((-1, 1), 1, p=[1 - prb, prb])
    return s_new


def new_node_vis(pixel, visible, hidden, W, V, bv):
    vis_neighbor = visible[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    vis_neighbor[1, 1] = 0
    hid_neighbor = hidden[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    prb = 1 / (1 + np.exp(-(np.sum(np.multiply(vis_neighbor, W)) +
                            np.sum(np.multiply(hid_neighbor, V)) + bv)))
    s_new = np.random.choice((-1, 1), 1, p=[1 - prb, prb])
    return s_new


def der_ret(pixel, S, pic):
    win = pic[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    return S * win


def BM_with_hidden(x_idx, x_prb,# class_mat, class_mat_border,
                   bm_epoch=1,
                   batch_size=300, learning_rate=.001, save_img=True):
    x_idx[x_idx == 0] = -1
    org_vis = x_idx.copy()
    org_hid = x_idx.copy()
    vis2 = x_idx.copy()

    wv2h = np.random.normal(0, 0.01, (3, 3))
    wh2v = np.random.normal(0, 0.01, (3, 3))
    whv = np.mean((wh2v, wv2h))
    wv2h[1, 1] = whv
    wh2v[1, 1] = whv
    v_prb = np.random.normal(0, 0.01, (3, 3))
    v = np.random.normal(0, 0.01, (3, 3))
    v[1, 1] = 0
    bh = np.array(0, dtype='float64')
    bv = np.array(0, dtype='float64')

    l = np.shape(x_idx)
#    print('l:', l)
    pix_pos = [(i, j) for i, j in np.ndindex((l[0]-2, l[1]-2))]

    x_place = [i for i in range((l[0]-2)*(l[1]-2))]

    n_batch = len(pix_pos) // batch_size + 1

    iter_ = data_iterator(np.add(np.array(pix_pos), 1), np.array(x_place), n_batch)

    for ep in tqdm(range(bm_epoch)):
        for ba in range(n_batch):
            vis = org_vis.copy()
            hid = org_hid.copy()

            pix, pos = next(iter_)

            # d_bh = np.array(0)
            # d_bv = np.array(0)
            # d_wv2h = np.zeros((3, 3))
            # d_wh2v = np.zeros((3, 3))
            # d_vp = np.zeros((3, 3))
            # d_v = np.zeros((3, 3))

            s_hid1 = np.array(list(map(lambda x: new_node_hid(x, vis, x_prb, wv2h, v_prb, bh), pix)))

            hid[pix[:, 0], pix[:, 1]] = s_hid1.reshape(np.shape(hid[pix[:, 0], pix[:, 1]]))

            s_vis1 = np.array(list(map(lambda x: new_node_vis(x, vis, hid, wh2v, v, bv), pix)))

            org_vis[pix[:, 0], pix[:, 1]] = s_vis1.reshape(np.shape(vis[pix[:, 0], pix[:, 1]]))

            s_hid2 = np.array(list(map(lambda x: new_node_hid(x, org_vis, x_prb, wv2h, v_prb, bh), pix)))

            org_hid[pix[:, 0], pix[:, 1]] = s_hid2.reshape(np.shape(hid[pix[:, 0], pix[:, 1]]))

            # s_vis2 = np.array(list(map(lambda x: new_node_vis(x, vis2, org_hid, wh2v, v, bv), pix)))
            #
            # org_vis[pix[:, 0], pix[:, 1]] = s_vis2.reshape(np.shape(vis[pix[:, 0], pix[:, 1]]))

            d_bv = np.sum(vis[pix[:, 0], pix[:, 1]] - org_vis[pix[:, 0], pix[:, 1]])
            d_bh = np.sum(hid[pix[:, 0], pix[:, 1]] - org_hid[pix[:, 0], pix[:, 1]])

            # d_v_pos = np.array(list(map(lambda x, y: der_ret(x, y, vis), pix, vis[pix[:, 0], pix[:, 1]])))
            # d_v_neg = np.array(list(map(lambda x, y: der_ret(x, y, org_vis), pix, org_vis[pix[:, 0], pix[:, 1]])))
            d_v = np.mean(list(map(lambda x, y: der_ret(x, y, vis), pix, org_vis[pix[:, 0], pix[:, 1]])), axis =0)
            d_v[1, 1] = 0
            # d_vp_pos = np.array(list(map(lambda x, y: der_ret(x, y, x_prb), pix, hid[pix[:, 0], pix[:, 1]])))
            # d_vp_neg = np.array(list(map(lambda x, y: der_ret(x, y, x_prb), pix, org_hid[pix[:, 0], pix[:, 1]])))
            d_vp = np.mean(list(map(lambda x, y: der_ret(x, y, x_prb), pix, org_hid[pix[:, 0], pix[:, 1]])), axis=0)
            # d_wv2h_pos = np.array(list(map(lambda x, y: der_ret(x, y, vis), pix, hid[pix[:, 0], pix[:, 1]])))
            # d_wv2h_neg = np.array(list(map(lambda x, y: der_ret(x, y, vis2), pix, org_hid[pix[:, 0], pix[:, 1]])))
            d_wv2h = np.mean(list(map(lambda x, y: der_ret(x, y, vis), pix, hid[pix[:, 0], pix[:, 1]])), axis=0)
            # d_wh2v_pos = np.array(list(map(lambda x, y: der_ret(x, y, hid), pix, vis2[pix[:, 0], pix[:, 1]])))
            # d_wh2v_neg = np.array(list(map(lambda x, y: der_ret(x, y, org_hid), pix, org_vis[pix[:, 0], pix[:, 1]])))
            d_wh2v = np.mean(list(map(lambda x, y: der_ret(x, y, hid), pix, org_vis[pix[:, 0], pix[:, 1]])), axis=0)

            d_whv = np.mean(np.multiply(vis[pix[:, 0], pix[:, 1]], hid[pix[:, 0], pix[:, 1]]) -
                            np.multiply(org_vis[pix[:, 0], pix[:, 1]], org_hid[pix[:, 0], pix[:, 1]]))
            d_wh2v[1, 1] = d_whv
            d_wv2h[1, 1] = d_whv

            # if ba == 1:
            #     print('d_wv2h_pos: ', d_wv2h_pos[0])
            #     print('d_wv2h_neg: ', d_wv2h_neg[0])
            #     print('d_wh2v_pos: ', d_wh2v_pos[0])
            #     print('d_wh2v_neg: ', d_wh2v_neg[0])
            wv2h += learning_rate * d_wv2h
            wh2v += learning_rate * d_wh2v
            v_prb += learning_rate * d_vp
            v += learning_rate * d_v
            bh += learning_rate * d_bh
            bv += learning_rate * d_bv

            # ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
            #
            # print('epoch num: ', ep, ' batch num: ', ba, 'accuracy value:', ac)


        if save_img == True:
            cv2.imwrite("Images/BM_hid/vis_"+str(ep)+".png", (org_vis + 1) * 255/2)
            cv2.imwrite("Images/BM_hid/hid_" + str(ep) + ".png", (org_hid + 1) * 255/2)
            plt.figure(1)
            plt.subplot(121)
            plt.imshow(org_vis)

            plt.subplot(122)
            plt.imshow(org_hid)
            plt.show()
    #
    #     cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(org_vis))
    #     print('confucion matrix value:', cm)
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print('confucion matrix value:', cm)
    #
    #
    #     ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
    #
    #     print('accuracy value:', ac)
    #
    # if save_img == True:
    #     image_show_border("Images/conv_whole_img/vis_"+str(ep)+".png", class_mat_border,
    #                       "Images/conv_whole_img/mumvis_"+str(ep)+".png")
    print('wh2v:', wh2v)
    print('wv2h:', wv2h)
    print('v:', v)
    print('vpr:', v_prb)
    print('bh:', bh)
    print('bv:', bv)

    return wv2h, wh2v, v_prb, v, bh, bv, org_vis


BM_with_hidden(img_idx, img_org,  # class_mat, class_mat_border,
                                                   bm_epoch=100,
                                                   batch_size=100, learning_rate=.001, save_img=True)

#
# def BM_hid_for(x_idx, x_pred, class_mat, class_mat_border, bm_epoch=20,
#                batch_size=128, learning_rate=.001, save_img=True):
#     # x_idx[x_idx == 0] = -1
#     vis = x_idx.copy()
#     hid = x_idx.copy()
#
#     wv2h = np.random.normal(0, 0.01, (3, 3))
#     wh2v = np.random.normal(0, 0.01, (3, 3))
#     whv = np.mean((wh2v, wv2h))
#     wv2h[1, 1] = whv
#     wh2v[1, 1] = whv
#     v_prb = np.random.normal(0, 0.01, (3, 3))
#     v = np.random.normal(0, 0.01, (3, 3))
#     v[1, 1] = 0
#     bh = np.array(0, dtype='float64')
#     bv = np.array(0, dtype='float64')
#
#     l = np.shape(x_pred)
#     #    print('l:', l)
#     pix_pos = [(i, j) for i, j in np.ndindex((l[0] - 2, l[1] - 2))]
#
#     x_place = [i for i in range((l[0] - 2) * (l[1] - 2))]
#
#     n_batch = len(pix_pos) // batch_size + 1
#
#     iter_ = data_iterator(np.add(np.array(pix_pos), 1), np.array(x_place), n_batch)
#
#     for ep in range(bm_epoch):
#         for ba in range(n_batch):
#             pix, _ = next(iter_)
#             d_bh = np.array(0)
#             d_bv = np.array(0)
#             d_wv2h = np.zeros((3, 3))
#             d_wh2v = np.zeros((3, 3))
#             d_vp = np.zeros((3, 3))
#             d_v = np.zeros((3, 3))
#             for pix_count in range(np.shape(pix)[0]):
#                 V0 = vis[pix[pix_count]]
#                 s_new = new_node_hid(pix[pix_count], vis, x_prb, wv2h, v_prb, bh)
#
#                 vis[x_img, y_img] = s_new[0]
#                 # print('d_b: ', d_b)
#                 d_b = d_b + np.array(S - s_new[0])
#                 d_w1 += delta_w(vis_win, s_new[0])
#                 d_w2 += delta_w(idx_win, s_new[0])
#                 d_w3 += delta_w(prb_win, s_new[0])
#
#             # print('db: ', np.array(d_b))
#             # print('np.shape(pix)[0]: ', float(np.shape(pix)[0]))
#             b1 = b1 + learning_rate * d_b / float(np.shape(pix)[0])
#             W1 += learning_rate * d_w1 / (np.shape(pix)[0])
#             W2 += learning_rate * d_w2 / (np.shape(pix)[0])
#             W3 += learning_rate * d_w3 / (np.shape(pix)[0])
#
#             vis0 = vis.copy()
#             vis0[vis0 == -1] = 0
#
#             ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
#
#             print('epoch num: ', ep, ' batch num: ', ba, 'accuracy value:', ac)
#
#         if save_img == True:
#             cv2.imwrite("Images/conv_whole_img/vis_" + str(ep) + ".png", vis0 * 255)
#
#         cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
#         print('confucion matrix value:', cm)
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print('confucion matrix value:', cm)
#
#         ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
#
#         print('accuracy value:', ac)
#
#     if save_img == True:
#         image_show_border("Images/conv_whole_img/vis_" + str(ep) + ".png", class_mat_border,
#                           "Images/conv_whole_img/mumvis_" + str(ep) + ".png")
#
#     return W1, W2, W3, b1, vis0

