#import matplotlib
#matplotlib.use('agg')
import cv2
import numpy as np
#from spyder_window_maker import data_iterator
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.io as sio

hm_epoch = 300
hm_iter = 3
lr = .001
save_address = "Images/BM_hid2/"

#
# mat_contents = sio.loadmat('lab_mat_wo_600')
# y = mat_contents['labeled_mat']
# y[y == 4] = 1
# class_mat = y[49:550, 49:550]
# y[y == 2] = 0
mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

y = mat_contents['B'][0][0][0][33]

y[y == 4] = 1
class_mat = y[49:49 + 201, 49:49 + 201]
class_mat2 = class_mat.copy()
class_mat[class_mat == 2] = 0


img_idx = cv2.imread('x_idx.png', 0)/255
img_org = cv2.imread('x_pr.png', 0)/255



def new_node_hid(pixel, visible, img_prb, W, V_prb, bh, t):
    vis_neighbor = visible[pixel[0]-1:pixel[0]+2, pixel[1]-1:pixel[1]+2]

    prb_neighbor = img_prb[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]

    prb = 1 / (1 + np.exp(-(np.sum(np.multiply(vis_neighbor, W)) +
                            np.sum(np.multiply(prb_neighbor, V_prb)) + bh)/t))
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


def der_ret(pixel, S, pic):
    win = pic[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
    return S * win


def BM_with_hidden(x_idx, x_prb,# class_mat, class_mat_border,
                   bm_epoch=1,
                   hm_iteration=10, learning_rate=.001, save_img=True):
    # x_idx[x_idx == 0] = -1

    w = np.random.normal(0, 0.01, (3, 3))
    v_prb = np.random.normal(0, 0.01, (3, 3))
    v = np.random.normal(0, 0.01, (3, 3))
    v[1, 1] = 0
    bh = np.array(0, dtype='float64')
    bv = np.array(0, dtype='float64')

    l = np.shape(x_idx)
#    print('l:', l)
    pix_pos = [(i, j) for i, j in np.ndindex((l[0]-2, l[1]-2))]
    pix = np.add(np.array(pix_pos), 1)

#    x_place = [i for i in range((l[0]-2)*(l[1]-2))]
    #
    # n_batch = len(pix_pos) // batch_size + 1
    #
    # iter_ = data_iterator(np.add(np.array(pix_pos), 1), np.array(x_place), n_batch)
    ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(x_idx))
    print('accuracy of x_idx: ', ac )
    for ep in tqdm(range(bm_epoch)):
        np.save('w', w)
        np.save('v', v)
        np.save('v_prb', v_prb)
        np.save('bv', bv)
        np.save('bh', bh)
        T = 1/(1.2 ** hm_epoch)
        org_vis = x_idx.copy()
        org_hid = np.zeros(np.shape(org_vis))
        vis = np.zeros(np.shape(org_vis))
        hid = np.zeros(np.shape(org_vis))
        for ba in range(hm_iteration):

            s_hid1 = np.array(list(map(lambda x: new_node_hid(
                                x, org_vis, x_prb, w, v_prb, bh, T), pix)))

            org_hid[pix[:, 0], pix[:, 1]] = s_hid1.reshape(np.shape(
                                                    hid[pix[:, 0], pix[:, 1]]))
            if ba == 0:
                hid = org_hid.copy()
                vis = org_vis.copy()

            else:

                s_vis1 = np.array(list(map(lambda x: new_node_vis(x, vis, hid, w,
                                                                  v, bv, T), pix)))

                vis[pix[:, 0], pix[:, 1]] = s_vis1.reshape(
                    np.shape(vis[pix[:, 0], pix[:, 1]]))

                s_hid2 = np.array(list(map(lambda x: new_node_hid(
                    x, vis, x_prb, w, v_prb, bh, T), pix)))

                hid[pix[:, 0], pix[:, 1]] = s_hid2.reshape(np.shape(
                    hid[pix[:, 0], pix[:, 1]]))

        d_bv = np.mean(org_vis[pix[:, 0], pix[:, 1]] - vis[pix[:, 0], pix[:, 1]])
        d_bh = np.mean(org_hid[pix[:, 0], pix[:, 1]] - hid[pix[:, 0], pix[:, 1]])

        d_v_neg = np.array(list(map(lambda x, y: der_ret(x, y, vis), pix, vis[pix[:, 0], pix[:, 1]])))
        d_v_pos = np.array(list(map(lambda x, y: der_ret(x, y, org_vis), pix, org_vis[pix[:, 0], pix[:, 1]])))
        d_v = np.mean(d_v_pos - d_v_neg, axis=0)
        d_v[1, 1] = 0
        d_vp_neg = np.array(list(map(lambda x, y: der_ret(x, y, x_prb), pix, hid[pix[:, 0], pix[:, 1]])))
        d_vp_pos = np.array(list(map(lambda x, y: der_ret(x, y, x_prb), pix, org_hid[pix[:, 0], pix[:, 1]])))
        d_vp = np.mean(d_vp_pos - d_vp_neg, axis=0)
        d_w1_neg = np.array(list(map(lambda x, y: der_ret(x, y, hid), pix, vis[pix[:, 0], pix[:, 1]])))
        d_w1_pos = np.array(list(map(lambda x, y: der_ret(x, y, org_hid), pix, org_vis[pix[:, 0], pix[:, 1]])))
        d_w1 = np.mean(d_w1_pos - d_w1_neg, axis=0)
        d_w2_neg = np.array(list(map(lambda x, y: der_ret(x, y, vis), pix, hid[pix[:, 0], pix[:, 1]])))
        d_w2_pos = np.array(list(map(lambda x, y: der_ret(x, y, org_vis), pix, org_hid[pix[:, 0], pix[:, 1]])))
        d_w2 = np.mean(d_w2_pos - d_w2_neg, axis=0)
        d_w = np.mean((d_w1, d_w2))

        # if ba == 1:
        #     print('d_wv2h_pos: ', d_wv2h_pos[0])
        #     print('d_wv2h_neg: ', d_wv2h_neg[0])
        #     print('d_wh2v_pos: ', d_wh2v_pos[0])
        #     print('d_wh2v_neg: ', d_wh2v_neg[0])
        w += learning_rate * d_w
        v_prb += learning_rate * d_vp
        v += learning_rate * d_v
        bh += learning_rate * d_bh
        bv += learning_rate * d_bv

        # ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
        #
        # print('epoch num: ', ep, ' batch num: ', ba, 'accuracy value:', ac)

        ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis))
        print('accuracy of vis in loop: ', ac )  
        ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(org_hid))
        print('accuracy of org_hid in loop: ', ac )
        ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(hid))
        print('accuracy of hid in loop: ', ac )
        if save_img == True:
            # cv2.imwrite("Images/BM_hid/vis_"+str(ep)+".png", (org_vis + 1) * 255/2)
            # cv2.imwrite("Images/BM_hid/hid_" + str(ep) + ".png", (org_hid + 1) * 255/2)
            plt.figure(1)
            plt.subplot(221)
            plt.imshow(org_vis)

            plt.subplot(222)
            plt.imshow(org_hid)

            plt.subplot(223)
            plt.imshow(vis)

#            plt.subplot(224)
#            plt.imshow(hid)
            
            new_vis = vis.copy()
            new_vis[class_mat2 == 2] = 2
            plt.subplot(224)
            plt.imshow(new_vis)
            plt.savefig(save_address + "all4_"+str(ep)+".png")
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
    # print('wh2v:', wh2v)
    # print('wv2h:', wv2h)
    # print('v:', v)
    # print('vpr:', v_prb)
    # print('bh:', bh)
    # print('bv:', bv)
    print('INPUT')
    cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(x_idx))
    print('confucion matrix value:', cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confucion matrix value:', cm)
    
    
    ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(x_idx))
    print('accuracy: ', ac )
    
    
    
    print('OUT_VIS')
    cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis))
    print('confucion matrix value:', cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confucion matrix value:', cm)
    
    
    ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis))
    print('accuracy: ', ac )

    return w, v_prb, v, bh, bv


w, v_prb, v, bh, bv = BM_with_hidden(img_idx, img_org,# class_mat, class_mat_border,
                   bm_epoch=hm_epoch,
                   hm_iteration=hm_iter, learning_rate=lr, save_img=True)

