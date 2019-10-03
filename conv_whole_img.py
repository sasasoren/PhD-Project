from spyder_window_maker import win_ftset_and_label, image_show_border, vote_filter, data_iterator
from MLP_after_auto import neural2
import scipy.io as sio
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 2
epochs = 1000
bol_epoch = 15
start_point = 49
window_size = 1
learning_rate = 0.001
mon_freq = 100
test_num = 5
# input image dimensions
img_rows, img_cols = 301, 301  #2 * window_size + 1, 2 * window_size + 1


img_num = [0, 7, 18, 33, 43, 58]


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

img_cut = img[start_point:start_point + 201, start_point:start_point + 201]

inputs_ = tf.placeholder(tf.float32, (None, img_rows, img_cols, 1), name="input")
targets_ = tf.placeholder(tf.float32, (None, img_rows, img_cols, 1), name="target")
keep_prob = tf.placeholder('float')

img_tr = np.array(img).reshape(1, 301, 301, 1)
img_tr_sec = np.array(img_sec).reshape(1, 301, 301, 1)

# Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7*7*8
# conv4 = tf.layers.conv2d(inputs=upsample1, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
logits = tf.layers.conv2d(inputs=conv1, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    j = 0
    for e in range(epochs):
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: img_tr,
                                                         targets_: img_tr,
                                                         keep_prob: .2})

        if e % mon_freq == 0:
            j += 1
            # summ = sess.run(summarymerged, feed_dict={inputs_: imgs, targets_: imgs})
            # writer.add_summary(summ, j)
            print("epoch = ", e, "batch loss = ", batch_cost)

    c1 = sess.run(conv1, feed_dict={inputs_: img_tr, targets_: img_tr, keep_prob: 1.0})
    c1 = c1[0, :, :, :]
    con1 = c1[start_point:start_point + 201, start_point:start_point + 201, :]

    c1_sec = sess.run(conv1, feed_dict={inputs_: img_tr_sec, targets_: img_tr_sec, keep_prob: 1.0})
    c1_sec = c1_sec[0, :, :, :]
    con1_sec = c1_sec[start_point:start_point + 201, start_point:start_point + 201, :]

x_train, y_train, x_test, y_test, x_, y_, label_x = win_ftset_and_label(c1, con1, class_mat,
                                                                                test_size=.1, win_type='3d',
                                                                                win_size=window_size,
                                                                                class_num=num_classes)
_, _, _, _, x_sec, y_sec, _ = win_ftset_and_label(c1_sec, con1_sec, class_mat_sec,
                                                                                test_size=.1, win_type='3d',
                                                                                win_size=window_size,
                                                                                class_num=num_classes)

x_pred, acc, conf, x_idx, x_idx_sec, x_pred_sec = neural2(x_train.astype('float32'),
                                                            y_train.astype('float32'),
                                                            x_test.astype('float32'),
                                                            y_test.astype('float32'),
                                                            x_.astype('float32'), other_test=x_sec,
                                                            hm_epochs=100, n_nodes_hl1=50, n_classes=2)

x_idx = x_idx.reshape(img_cut.shape)
x_idx, counter = vote_filter(x_idx, thrshold=7)
cv2.imwrite("Images/conv_whole_img/x_idx.png", x_idx*255)


x_pred = x_pred[:, 0]

x_pred = x_pred.reshape(img_cut.shape)
# print('img_cut_shape:', img_cut.shape)
# print('x_pred shape:', x_pred.shape)


# def function for WX+b
def con_bolt(vis, hid1, hid2, w1, w2, w3, b1):

    """
    vis:    8*1 nwighbor of pixel
    hid1:   9*1 neighbors of idx layer
    hid1:   9*1 neighbors of prob layer
    w1, w2, w3:  correspondent weigh of vis, hid1 and hid3
    b1 :    biases for vis
    we are returning WX+b which is a number
    """
    # print('visw1: ', np.matmul(vis, w1))
    # print('b1:', b1)
    # print('hid1w2: ', np.matmul(hid1, w2))
    # print('hid2w3: ', np.matmul(hid2, w3))
    wxb = np.matmul(vis, w1) + b1 + np.matmul(hid1, w2) + np.matmul(hid2, w3)
    # print('wxb: ', wxb)
    return wxb

# def delta_b(vis_i, vis_i1):
#     """
#     vis_i:
#     vis_i1:
#     output:  vector delta_b which is a correspondent biases for each neighbours.
#     """
#     d_b = np.sum(vis_i1) - np.sum(vis_i)
#     return d_b

def delta_w(neighbours, s):
    """
    neighbours:   all neighbours of pixel in each layer layer
    s:            the value of pixel
    output:       vector delta_w which is correspondent with related weight
    """
    d_w = neighbours * s
    return d_w


def BM_scratch(x_idx, x_pred, class_mat, class_mat_border, bm_epoch=20, batch_size=batch_size, save_img=True):
    x_idx[x_idx == 0] = -1
    vis = x_idx.copy()

    W1 = np.random.normal(0, 0.01, 8)
    W2 = np.random.normal(0, 0.01, 9)
    W3 = np.random.normal(0, 0.01, 9)
    b1 = np.array(0)

    l = np.shape(x_pred)
    # print('l:', l)
    pix_pos = [(i, j) for i, j in np.ndindex((l[0]-2, l[1]-2))]

    x_place = [i for i in range((l[0]-2)*(l[1]-2))]

    n_batch = len(pix_pos) // batch_size + 1

    iter_ = data_iterator(np.add(np.array(pix_pos),1), np.array(x_place), n_batch)

    for ep in range(bm_epoch):
        for ba in range(n_batch):
            pix, _ = next(iter_)
            d_b = np.array(0)
            d_w1 = np.zeros(8)
            d_w2 = np.zeros(9)
            d_w3 = np.zeros(9)
            for pix_count in range(np.shape(pix)[0]):
                x_img, y_img = pix[pix_count]
                vis_pix = vis[x_img-1:x_img+2, y_img-1:y_img+2]
                vis_win = list(np.ndarray.flatten(vis_pix))
                S = np.array(vis_win[4])
                # print('S: ', S)
                del vis_win[4]
                vis_win = np.array(vis_win)
                # print('vis_win: ', vis_win)
                idx_win = np.ndarray.flatten(x_idx[x_img-1:x_img+2, y_img-1:y_img+2])
                # print('idx_win: ', idx_win)
                prb_win = np.ndarray.flatten(x_pred[x_img - 1:x_img + 2, y_img - 1:y_img + 2])
                # print('prb_win: ', prb_win)
                pix_wx = con_bolt(vis_win, idx_win, prb_win, W1, W2, W3, b1)
                # print('pix_wx: ', pix_wx)
                prb_pix = 1/(1+np.exp(-pix_wx))
                # print('prb_pix: ', prb_pix)
                # print(ba, pix_count, x_img, y_img)
                s_new = np.random.choice((-1, 1), 1, p=[1-prb_pix, prb_pix])
                # print('s_new:', s_new[0])
                vis[x_img, y_img] = s_new[0]
                # print('d_b: ', d_b)
                d_b += np.array(S - s_new[0])
                d_w1 += delta_w(vis_win, s_new[0])
                d_w2 += delta_w(idx_win, s_new[0])
                d_w3 += delta_w(prb_win, s_new[0])

            # print('db: ', np.array(d_b))
            # print('np.shape(pix)[0]: ', float(np.shape(pix)[0]))
            b1 = b1 + learning_rate * d_b/float(np.shape(pix)[0])
            W1 += learning_rate * d_w1/(np.shape(pix)[0])
            W2 += learning_rate * d_w2/(np.shape(pix)[0])
            W3 += learning_rate * d_w3/(np.shape(pix)[0])

            vis0 = vis.copy()
            vis0[vis0 == -1] = 0

            ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))

            print('epoch num: ', ep, ' batch num: ', ba, 'accuracy value:', ac)

        if save_img == True:
            cv2.imwrite("Images/conv_whole_img/vis_"+str(ep)+".png", vis0 * 255)

        cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
        print('confucion matrix value:', cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confucion matrix value:', cm)


        ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))

        print('accuracy value:', ac)

    if save_img == True:
        image_show_border("Images/conv_whole_img/vis_"+str(ep)+".png", class_mat_border,
                          "Images/conv_whole_img/mumvis_"+str(ep)+".png")
    if ac < .3:
        print('Accuracy is decreasing so we have multiply weights and biases with negative one')
        W1 = -W1
        W2 = -W2
        W3 = -W3
        b1 = -b1

    return W1, W2, W3, b1, vis0, ac, cm


W1, W2, W3, b1, vis0, accu, conm = BM_scratch(x_idx, x_pred,  class_mat, class_mat2, bm_epoch=15)


def BM_predict(x_idx, x_pred, W1, W2, W3, b1, class_mat, class_mat_border, bm_epoch=20, error=.001,
               batch_size=batch_size, save_img=True):
    x_idx[x_idx == 0] = -1
    vis = x_idx.copy()

    l = np.shape(x_pred)
    print('l:', l)
    pix_pos = [(i, j) for i, j in np.ndindex((l[0] - 2, l[1] - 2))]
    print('pix_pos: ',pix_pos[0])
    x_place = [i for i in range((l[0] - 2) * (l[1] - 2))]
    print('x_place: ',x_place[0])
    n_batch = len(pix_pos) // batch_size + 1
    print('np.add(np.array(pix_pos),1):',np.add(np.array(pix_pos),1)[0])
    iter2_ = data_iterator(np.add(np.array(pix_pos),1), np.array(x_place), n_batch)

    last_ac = 0

    for ep in range(bm_epoch):
        # print('epoch num:', ep)
        for ba in range(n_batch):
            # print('batch num', ba)
            pix, _ = next(iter2_)
            # print('pix0: ', pix[0])
            for pix_count in range(np.shape(pix)[0]):
                x_img, y_img = pix[pix_count]
                vis_pix = vis[x_img - 1:x_img + 2, y_img - 1:y_img + 2]
                # print('vis_pix:', vis_pix)
                vis_win = list(np.ndarray.flatten(vis_pix))
                # print('here, before del')
                del vis_win[4]
                vis_win = np.array(vis_win)
                idx_win = np.ndarray.flatten(x_idx[x_img - 1:x_img + 2, y_img - 1:y_img + 2])
                prb_win = np.ndarray.flatten(x_pred[x_img - 1:x_img + 2, y_img - 1:y_img + 2])
                pix_wx = con_bolt(vis_win, idx_win, prb_win, W1, W2, W3, b1)
                # print('after convolution')
                prb_pix = 1 / (1 + np.exp(-pix_wx))
                s_new = np.random.choice((-1, 1), 1, p=[1 - prb_pix, prb_pix])
                vis[x_img, y_img] = s_new[0]

            vis0 = vis.copy()
            vis0[vis0 == -1] = 0

            ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))

            print('epoch num: ', ep, ' batch num: ', ba, 'accuracy value:', ac)

        if save_img == True:
            cv2.imwrite("Images/conv_whole_img/vis_pred_" + str(ep) + ".png", vis0 * 255)

        cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))
        print('confucion matrix value:', cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confucion matrix value:', cm)

        ac = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(vis0))

        print('accuracy value:', ac)

        if abs(ac - last_ac) < error:
            print('Stop in epoch number: '+str(ep)+' error is less than: '+str(error))
            break
        last_ac = ac



    if save_img == True:
        image_show_border("Images/conv_whole_img/vis_pred_" + str(ep) + ".png", class_mat_border,
                          "Images/conv_whole_img/mumvis_pred_" + str(ep) + ".png")

x_idx_sec = x_idx_sec.reshape(img_cut.shape)
x_idx_sec, counter2 = vote_filter(x_idx_sec, thrshold=7)
cv2.imwrite("Images/conv_whole_img/x_idx_SEC.png", x_idx_sec*255)


x_pred_sec = x_pred_sec[:, 0]
x_pred_sec = x_pred_sec.reshape(img_cut.shape)
BM_predict(x_idx_sec, x_pred_sec, W1, W2, W3, b1, class_mat=class_mat_sec, class_mat_border=class_mat2_sec,
           bm_epoch=20, error=.001, save_img=True)

print('accuracy for training boltzmann: ', accu)
print('confusion matrix for training boltzmann: ', conm)

x_idx[x_idx == -1] = 0
ac1 = accuracy_score(np.ndarray.flatten(class_mat), np.ndarray.flatten(x_idx))
print(x_idx[0])
print(np.unique(class_mat[0]))
cm = confusion_matrix(np.ndarray.flatten(class_mat), np.ndarray.flatten(x_idx))
print('confucion matrix value:', cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('confucion matrix value:', cm)
print('accuracy1: ', ac1)
x_idx_sec[x_idx_sec == -1] = 0
ac2 = accuracy_score(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(x_idx_sec))
cm2 = confusion_matrix(np.ndarray.flatten(class_mat_sec), np.ndarray.flatten(x_idx_sec))
print('confucion matrix value:', cm2)
cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
print('confucion matrix value:', cm2)
print('accuracy1: ', ac2)


print('THE END')














































