from spyder_window_maker import win_ftset_and_label, data_iterator, image_show_border, vote_filter, BM_scratch, BM_predict
import scipy.io as sio
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from skimage.measure import label

test_num = 5
# input image dimensions

img_num = [0, 7, 18, 33, 43, 58]


mat_contents = sio.loadmat('mum-perf-org-new-1-60.mat')

y = mat_contents['B'][0][0][0][33]
img = mat_contents['B'][0][2][0][33]

y_sec = mat_contents['B'][0][0][0][img_num[test_num]]
img_sec = mat_contents['B'][0][2][0][img_num[test_num]]


def conv_fun(img, img_cls, img_sec, img_cls_sec, img_width,
             img_length, save_add, batch_size=128,
             num_classes=2, epochs=30, start_point=49,
             window_size=3, learning_rate=0.001,
             mon_freq=100, pr_win_size=1,
             pr_num_fil=0, n_nodes_hl1=7):

    # This function get these input and give you predicted output after
    # autoencoder with convolution layer and maxpool after that mlp layer
    # at the end we have pr_num_fil times convolution layer over output of
    # predicted of cnn.

    sys.stdout = open(save_add + "all_text.txt", "w")
    img_rows, img_cols = 2 * window_size + 1, 2 * window_size + 1

    img_cls[img_cls == 4] = 1
    img_cls_sec[img_cls_sec == 4] = 1
    class_mat = img_cls[start_point:start_point + img_width,
                start_point:start_point + img_length]
    class_mat2 = class_mat.copy()
    class_mat[class_mat == 2] = 0
    class_mat_sec = img_cls_sec[start_point:start_point + img_width,
                    start_point:start_point + img_length]
    class_mat2_sec = class_mat_sec.copy()
    class_mat_sec[class_mat_sec == 2] = 0

    img_cut = img[start_point:start_point + img_width,
              start_point:start_point + img_length]
    cv2.imwrite(save_add + "org1.png", img_cut*255)
    image_show_border(save_add + "org1.png", class_mat2,
                      save_add + "mumorg1.png")
    img_cut_sec = img_sec[start_point:start_point + img_width,
                          start_point:start_point + img_length]
    cv2.imwrite(save_add + "org_test.png", img_cut_sec*255)
    image_show_border(save_add + "org_test.png", class_mat2_sec, save_add + "mumorg_test.png")

    x_train, y_train, x_test, y_test, x_, y_,\
    label_x = win_ftset_and_label(img, img_cut, class_mat,
                                  test_size=.1,
                                  win_size=window_size,
                                  class_num=num_classes)

    x_train_sec, y_train_sec, x_test_sec, y_test_sec,\
    xsec_, ysec_, _ = win_ftset_and_label(img_sec, img_cut_sec,
                                          class_mat_sec,
                                          test_size=.1,
                                          win_size=window_size,
                                          class_num=num_classes)

    # Input and target placeholders
    inputs_ = tf.placeholder(tf.float32, (
        None, img_rows, img_cols, 1), name="input")
    targets_ = tf.placeholder(tf.float32, (
        None, img_rows, img_cols, 1), name="target")

    # Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=8,
                             kernel_size=(3, 3), padding='same',
                             activation=tf.nn.relu)
    # Now 7*7*8
    # maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    encoded = tf.layers.max_pooling2d(conv1, pool_size=(2, 2),
                                      strides=(2, 2),
                                      padding='same')
    # Now 4x4x8
    # conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # # Now 14x14x8
    # maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # # Now 7x7x8
    # conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # # Now 7x7x8
    # encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # # Now 4x4x8

    ### Decoder
    upsample1 = tf.image.resize_images(encoded, size=(7, 7),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x8
    # conv4 = tf.layers.conv2d(inputs=upsample1, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    logits = tf.layers.conv2d(inputs=upsample1, filters=1,
                              kernel_size=(3, 3),
                              padding='same',
                              activation=tf.nn.relu)
    # Now 7x7x1

    # upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # # Now 14x14x8
    # conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # # Now 14x14x8
    # upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # # Now 28x28x8
    # conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # # Now 28x28x16

    # logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
    #Now 28x28x1

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    # Pass logits through sigmoid and calculate the cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_,
                                                   logits=logits)

    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    iter_ = data_iterator(x_train, y_train, batch_size)

    n_batch = len(x_train) // batch_size + 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(n_batch):
                batch_x, batch_y = next(iter_)
                imgs = batch_x.reshape((-1, img_rows, img_cols, 1))
                batch_cost, _ = sess.run([cost, opt],
                                         feed_dict={inputs_: imgs,
                                                    targets_: imgs})

                if ii % mon_freq == 0:
                    print("epoch = ", e, "iteration = ", ii,
                          "batch loss = ", batch_cost)

        x_train_new = x_train.reshape((-1, img_rows, img_cols, 1))
        x_train_n = sess.run(encoded, feed_dict={inputs_: x_train_new,
                                                 targets_: x_train_new})

        x_new = x_.reshape((-1, img_rows, img_cols, 1))
        x_n = sess.run(encoded, feed_dict={inputs_: x_new,
                                           targets_: x_new})

        x_new_sec = xsec_.reshape((-1, img_rows, img_cols, 1))
        x_n_sec = sess.run(encoded, feed_dict={inputs_: x_new_sec,
                                               targets_: x_new_sec})

        x_test_new = x_test.reshape((-1, img_rows, img_cols, 1))
        x_test_n = sess.run(encoded,
                            feed_dict={inputs_: x_test_new,
                                       targets_: x_test_new})

    x_h = tf.placeholder(tf.float32, (None, 4, 4, 8))
    y_all = tf.placeholder(tf.float32)

    def cn_net(x_h):
        conv2 = tf.layers.conv2d(inputs=x_h, filters=32,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation=tf.nn.relu)

        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 50)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, num_classes)
        return out


    iter2_ = data_iterator(x_train_n, y_train, batch_size)

    x_pred = tf.nn.sigmoid(cn_net(x_h))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=x_pred, labels=y_all))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    pred_idx = tf.argmax(x_pred, 1)
    y_idx = tf.argmax(y_all, 1)
    correct = tf.equal(pred_idx, y_idx)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    confusion = tf.confusion_matrix(labels=y_idx,
                                    predictions=pred_idx,
                                    num_classes=num_classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter2_)
                if i % mon_freq == 0:
                    batch_loss = sess.run(
                        cost, feed_dict={x_h: batch_x,
                                         y_all: batch_y})

                    print("epoch = ", epoch, "iteration = ",
                          i, "batch loss = ", batch_loss)

                sess.run(optimizer, feed_dict={x_h: batch_x,
                                               y_all: batch_y})

        print('Accuracy:', accuracy.eval({x_h: x_test_n,
                                          y_all: y_test}))
        print('Accuracy_:', accuracy.eval({x_h: x_n, y_all: y_}))
        print('Accuracy_sec:', accuracy.eval({x_h: x_n_sec,
                                              y_all: ysec_}))
        # acc = accuracy.eval({x_h: x_test_n, y_all: y_test})
        confu = sess.run(confusion, feed_dict={x_h: x_test_n,
                                               y_all: y_test})
        print('Confusion:', confu)
        confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
        print('Confusion_percentage:', confu)
        print('Confusion_:', sess.run(confusion,
                                      feed_dict={x_h: x_n,
                                                 y_all: y_}))
        confu = sess.run(confusion, feed_dict={x_h: x_n_sec,
                                               y_all: ysec_})
        print('Confusion_sec:', confu)
        confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
        print('Confusion_sec_percentage:', confu)
        # conf = sess.run(confusion, feed_dict={x_h: x_test_n, y_all: y_test})
        x_idx = sess.run(pred_idx, feed_dict={x_h: x_n})
        x_pr = sess.run(x_pred, feed_dict={x_h: x_n})[:, 0]\
            .reshape(img_cut.shape)
        cv2.imwrite("x_pr.png", x_pr * 255)
        img_bin, counter1 = vote_filter(x_idx.reshape(img_cut.shape),
                                        window_size=1, thrshold=7)
        print('counter1:', counter1)
        # print('img_bin.unique', np.unique(img_bin))
        cv2.imwrite(save_add + "CNN1.png", img_bin*255)
        image_show_border(save_add + "CNN1.png", class_mat2,
                          save_add + "mumCNN1.png")
        x_idx_sec = sess.run(pred_idx, feed_dict={x_h: x_n_sec})
        x_pr_sec = sess.run(x_pred, feed_dict={x_h: x_n_sec})[:, 0].\
            reshape(img_cut.shape)
        cv2.imwrite("x_pr_sec.png", x_pr_sec * 255)
        img_bin_sec, counter2 = vote_filter(
            x_idx_sec.reshape(img_cut.shape).reshape(img_cut.shape),
            window_size=1, thrshold=7)
        print('counter2:', counter2)
        # print('img_bin.unique', np.unique(img_bin))
        cv2.imwrite(save_add + "CNN_test.png", img_bin_sec*255)
        image_show_border(save_add + "CNN_test.png", class_mat2_sec,
                          save_add + "mumCNN_test.png")

    def neural_network_model(data, hidden_1_layer, output_layer):
        with tf.name_scope('Hidden_1'):
            l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
            l1 = tf.nn.relu(l1)
            tf.summary.histogram("weight_1", hidden_1_layer['weight'])

        with tf.name_scope('Output'):
            output = tf.matmul(l1, output_layer['weight']) + output_layer['bias']

        return output

    for num_fil in range(pr_num_fil):
        x_pr_pad = np.zeros(np.add(np.shape(x_pr), pr_win_size*2))
        x_pr_sec_pad = np.zeros(np.add(np.shape(x_pr_sec), pr_win_size * 2))
        x_pr_pad[pr_win_size:np.shape(x_pr_pad)[0]-pr_win_size,
        pr_win_size:np.shape(x_pr_pad)[1]-pr_win_size] = x_pr
        x_pr_sec_pad[pr_win_size:np.shape(x_pr_sec_pad)[0] - pr_win_size,
        pr_win_size:np.shape(x_pr_sec_pad)[1] - pr_win_size] = x_pr_sec

        x_train2, y_train2, x_test2, y_test2, x_2, y_2, \
        label_x2 = win_ftset_and_label(x_pr_pad, x_pr, class_mat,
                                       test_size=.1,
                                       win_size=pr_win_size,
                                       class_num=num_classes)

        x_train_sec2, y_train_sec2, x_test_sec2, y_test_sec2, \
        xsec_2, ysec_2, _ = win_ftset_and_label(x_pr_sec_pad, x_pr_sec,
                                                class_mat_sec,
                                                test_size=.1,
                                                win_size=pr_win_size,
                                                class_num=num_classes)

        x2 = tf.placeholder('float')
        y2 = tf.placeholder('float')

        n_batch = len(x_train2) // batch_size + 1

        hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(x_train2[0]), n_nodes_hl1]), name='w1'),
                          'bias': tf.Variable(tf.random_normal([n_nodes_hl1]), name='b1')}
        print(hidden_1_layer['weight'].name)
        output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, num_classes]), name='w2'),
                        'bias': tf.Variable(tf.random_normal([num_classes]), name='b2')}

        iter_2 = data_iterator(x_train2, y_train2, batch_size)

        prediction2 = neural_network_model(x2, hidden_1_layer, output_layer)
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction2, labels=y2))
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)

        pred_pr2 = tf.nn.sigmoid(prediction2)
        pred_idx2 = tf.argmax(prediction2, 1)
        y_idx2 = tf.argmax(y2, 1)
        correct2 = tf.equal(pred_idx2, y_idx2)
        accuracy2 = tf.reduce_mean(tf.cast(correct2, 'float'))

        confusion2 = tf.confusion_matrix(labels=tf.argmax(y2, 1), predictions=tf.argmax(prediction2, 1),
                                         num_classes=num_classes)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):

                for i2 in range(n_batch):
                    batch_x, batch_y = next(iter_2)
                    # if i2 % mon_freq == 0:
                    #
                    #     batch_loss = sess.run(cost2, feed_dict={x2: batch_x, y2: batch_y})
                    #
                    #     print("epoch = ", epoch, "iteration = ", i2, "batch loss = ", batch_loss)

                    sess.run(optimizer2, feed_dict={x2: batch_x, y2: batch_y})

            print('Accuracy2:', accuracy2.eval({x2: x_test2,
                                              y2: y_test2}))
            print('Accuracy_2:', accuracy2.eval({x2: x_2, y2: y_2}))
            print('Accuracy_sec2:', accuracy2.eval({x2: xsec_2,
                                                  y2: ysec_2}))

            confu2 = sess.run(confusion2, feed_dict={x2: x_test2,
                                                   y2: y_test2})
            print('Confusion2:', confu2)
            confu2 = confu2.astype('float') / confu2.sum(axis=1)[:, np.newaxis]
            print('Confusion_percentage2:', confu2)
            print('Confusion_2:', sess.run(confusion2,
                                           feed_dict={x2: x_2,
                                                      y2: y_2}))
            confu2 = sess.run(confusion2, feed_dict={x2: xsec_2,
                                                     y2: ysec_2})
            print('Confusion_sec2:', confu2)
            confu2 = confu2.astype('float') / confu2.sum(axis=1)[:, np.newaxis]
            print('Confusion_sec_percentage2:', confu2)
            # conf = sess.run(confusion, feed_dict={x_h: x_test_n, y_all: y_test})
            x_idx2 = sess.run(pred_idx2, feed_dict={x2: x_2})

            x_pr = sess.run(pred_pr2, feed_dict={x2: x_2})[:, 0] \
                .reshape(img_cut.shape)
            cv2.imwrite(save_add + "x_pr2"+str(num_fil) + ".png", x_pr * 255)
            # img_bin, counter1 = vote_filter(x_idx.reshape(img_cut.shape),
            #                                 window_size=1, thrshold=7)
            # print('counter1:', counter1)
            # print('img_bin.unique', np.unique(img_bin))
            cv2.imwrite(save_add + "CNN12.png", x_idx2.reshape(img_cut.shape) * 255)

            image_show_border(save_add + "CNN12.png", class_mat2,
                              save_add + "mumCNN12.png")
            x_idx_sec2 = sess.run(pred_idx2, feed_dict={x2: xsec_2})
            x_pr_sec = sess.run(pred_pr2, feed_dict={x2: xsec_2})[:, 0].reshape(img_cut.shape)

            cv2.imwrite(save_add + "x_pr_sec2" + str(num_fil) + ".png", x_pr_sec * 255)
            # img_bin_sec, counter2 = vote_filter(
            #     x_idx_sec.reshape(img_cut.shape).reshape(img_cut.shape),
            #     window_size=1, thrshold=7)
            # print('counter2:', counter2)
            # print('img_bin.unique', np.unique(img_bin))
            cv2.imwrite(save_add + "CNN_test2" + str(num_fil) + ".png", x_idx_sec2.reshape(img_cut.shape) * 255)
            image_show_border(save_add + "CNN_test2" + str(num_fil) + ".png", class_mat2_sec,
                              save_add + "mumCNN_test2" + str(num_fil) + ".png")

    sys.stdout.close()
    return x_idx2.reshape(img_cut.shape),\
           x_idx_sec2.reshape(img_cut.shape),\
           img_cut, img_cut_sec


def label_img(img_lab, img_bin, color_=(0, 0, 255), lab=None):
    if lab.all() == None:
        labeled_img = label(img_bin, connectivity=1)
    else:
        labeled_img = lab

    # print("Number of cells: ", np.max(labeled_img) + 1)
    for i in range(1, np.max(labeled_img) + 1):
        cord_ = np.where(labeled_img == i)
        y_center = int(np.sum(cord_[0]) / len(cord_[0])) + 1
        x_center = int(np.sum(cord_[1]) / len(cord_[1])) - 1
        cv2.putText(img_lab, str(i), (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 3 / 10, color_, 1)
    return img_lab


def col_cell(img, cell_points, color_=(0, 255, 255)):
    for i in range(len(cell_points)):
        cv2.circle(img, (cell_points[i, 0],
                         cell_points[i, 1]),
                   1, color_, 1)
        return img

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

    
def draw_flow(img, flow, step=16, point_=None):
    h, w = img.shape[:2]
    if point_ == None:
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    else:
        y, x = point_
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)

    if point_ == None:
        return vis
    else:
        return vis, lines

# conv_fun(img, y, img_sec, y_sec, img_width=201,
#          img_length=201, save_add="Images/conv_fun/")











