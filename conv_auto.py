from spyder_window_maker import win_ftset_and_label, image_show_border, vote_filter, BM_scratch, BM_predict
import scipy.io as sio
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


batch_size = 128
num_classes = 2
epochs = 20
start_point = 49
window_size = 3
learning_rate = 0.001
mon_freq = 100
test_num = 5
# input image dimensions
img_rows, img_cols = 2 * window_size + 1, 2 * window_size + 1

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
cv2.imwrite("Images/org1.png", img_cut*255)
image_show_border("Images/org1.png", class_mat,
                  "Images/mumorg1.png")
img_cut_sec = img_sec[start_point:start_point + 201, start_point:start_point + 201]
cv2.imwrite("Images/org" + str(img_num[test_num])+ ".png", img_cut_sec*255)
image_show_border("Images/org" + str(img_num[test_num])+ ".png", class_mat_sec,
                  "Images/mumorg" + str(img_num[test_num])+ ".png")
x_train, y_train, x_test, y_test, x_, y_, label_x = win_ftset_and_label(img, img_cut, class_mat,
                                                                        test_size=.1,
                                                                        win_size=window_size,
                                                                        class_num=num_classes)
x_train_sec, y_train_sec, x_test_sec, y_test_sec, xsec_, ysec_, _ = win_ftset_and_label(img_sec, img_cut_sec
                                                                                        , class_mat_sec,
                                                                                        test_size=.1,
                                                                                        win_size=window_size,
                                                                                        class_num=num_classes)
# showing the first window
# win_example = x_train[0]
# plt.imshow(win_example.reshape((img_rows, img_cols)), cmap='Greys_r')
# plt.show()

# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, img_rows, img_cols, 1), name="input")
targets_ = tf.placeholder(tf.float32, (None, img_rows, img_cols, 1), name="target")

# Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7*7*8
# maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
encoded = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
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
upsample1 = tf.image.resize_images(encoded, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x8
# conv4 = tf.layers.conv2d(inputs=upsample1, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
logits = tf.layers.conv2d(inputs=upsample1, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
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
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


def data_iterator(train_images, train_labels, batch_size):
    """ A simple data iterator """
    n = train_images.shape[0]
    # batch_idx = 0
    while True:

        shuf_idxs = np.random.permutation(n).reshape((1, n))[0]
        shuf_images = train_images[shuf_idxs]
        shuf_labels = train_labels[shuf_idxs]

        for batch_idx in range(0, n, batch_size):
            # print(shuf_idxs[batch_idx: batch_idx + batch_size])
            batch_images = shuf_images[batch_idx: batch_idx + batch_size]
            batch_labels = shuf_labels[batch_idx: batch_idx + batch_size]
            yield batch_images, batch_labels


iter_ = data_iterator(x_train, y_train, batch_size)

n_batch = len(x_train) // batch_size + 1

saver = tf.train.Saver()


with tf.Session() as sess:
    summarymerged = tf.summary.merge_all()
    filename = "summary_log"
    writer = tf.summary.FileWriter(filename, sess.graph)
    sess.run(tf.global_variables_initializer())
    j = 0
    for e in range(epochs):
        for ii in range(n_batch):
            batch_x, batch_y = next(iter_)
            imgs = batch_x.reshape((-1, img_rows, img_cols, 1))
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                             targets_: imgs})

            if ii % mon_freq == 0:
                j += 1
                # summ = sess.run(summarymerged, feed_dict={inputs_: imgs, targets_: imgs})
                # writer.add_summary(summ, j)
                print("epoch = ", e, "iteration = ", ii, "batch loss = ", batch_cost)

    x_train_new = x_train.reshape((-1, img_rows, img_cols, 1))
    x_train_n = sess.run(encoded, feed_dict={inputs_: x_train_new, targets_: x_train_new})

    x_new = x_.reshape((-1, img_rows, img_cols, 1))
    x_n = sess.run(encoded, feed_dict={inputs_: x_new, targets_: x_new})

    x_new_sec = xsec_.reshape((-1, img_rows, img_cols, 1))
    x_n_sec = sess.run(encoded, feed_dict={inputs_: x_new_sec, targets_: x_new_sec})

    x_test_new = x_test.reshape((-1, img_rows, img_cols, 1))
    x_test_n = sess.run(encoded, feed_dict={inputs_: x_test_new, targets_: x_test_new})

    for cls in range(2):
        img_s = x_train[np.where(y_train[:, cls] == 1)]
        # print('np.shape(img_s)', np.shape(img_s))
        ax = plt.subplot(2, 5, 1)
        img_show = img_s[100].reshape(img_cols, img_cols)
        plt.imshow(img_show)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 5, 10)
        in_ = img_s[100].reshape(1, img_cols, img_cols, 1)
        img_show = sess.run(logits, feed_dict={inputs_: in_, targets_: in_})
        img_show = img_show[0, :, :, 0].reshape(7, 7)
        plt.imshow(img_show)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for h in range(8):
            img_s = x_train_n[np.where(y_train[:, cls] == 1)]
            # print('np.shape(x_train_n)', np.shape(x_train_n))
            ax = plt.subplot(2, 5, h+2)
            # print('np.shape(img_s):', np.shape(img_s))
            # print('np.shape(img_s[h]):', np.shape(img_s[h]))
            img_show = img_s[100, :, :, h].reshape(4, 4)
            plt.imshow(img_show)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


        plt.savefig("Images/vis-hid"+str(cls)+".png")


x_h = tf.placeholder(tf.float32, (None, 4, 4, 8))
y_all = tf.placeholder(tf.float32)


def cn_net(x_h):
    # upsample2 = tf.image.resize_images(x_h, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv2 = tf.layers.conv2d(inputs=x_h, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)


    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv2)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 50)
    # # Apply Dropout (if is_training is False, dropout is not applied)
    # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(fc1, num_classes)
    return out

iter2_ = data_iterator(x_train_n, y_train, batch_size)


x_pred = tf.nn.sigmoid(cn_net(x_h))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x_pred, labels=y_all))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

pred_idx = tf.argmax(x_pred, 1)
y_idx = tf.argmax(y_all, 1)
correct = tf.equal(pred_idx, y_idx)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


confusion = tf.confusion_matrix(labels=y_idx, predictions=pred_idx, num_classes=num_classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    j=0
    #epoch_loss = 0
    for epoch in range(epochs):

        for i in range(n_batch):
            batch_x, batch_y = next(iter2_)
            if i % mon_freq == 0:
                j += 1
                batch_loss = sess.run(cost, feed_dict={x_h: batch_x, y_all: batch_y})

                print("epoch = ", epoch, "iteration = ", i, "batch loss = ", batch_loss)

            sess.run(optimizer, feed_dict={x_h: batch_x, y_all: batch_y})





    print('Accuracy:', accuracy.eval({x_h: x_test_n, y_all: y_test}))
    print('Accuracy_:', accuracy.eval({x_h: x_n, y_all: y_}))
    print('Accuracy_sec:', accuracy.eval({x_h: x_n_sec, y_all: ysec_}))
    # acc = accuracy.eval({x_h: x_test_n, y_all: y_test})
    confu = sess.run(confusion, feed_dict={x_h: x_test_n, y_all: y_test})
    print('Confusion:', confu)
    confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
    print('Confusion_percentage:', confu)
    print('Confusion_:', sess.run(confusion, feed_dict={x_h: x_n, y_all: y_}))
    confu = sess.run(confusion, feed_dict={x_h: x_n_sec, y_all: ysec_})
    print('Confusion_sec:', confu)
    confu = confu.astype('float') / confu.sum(axis=1)[:, np.newaxis]
    print('Confusion_sec_percentage:', confu)
    # conf = sess.run(confusion, feed_dict={x_h: x_test_n, y_all: y_test})
    x_idx = sess.run(pred_idx, feed_dict={x_h: x_n})
    x_pr = sess.run(x_pred, feed_dict={x_h: x_n})[:, 0].reshape(img_cut.shape)
    cv2.imwrite("x_pr.png", x_pr * 255)
    img_bin, counter1 = vote_filter(x_idx.reshape(img_cut.shape), window_size=1, thrshold=7)
    print('counter1:', counter1)
    # print('img_bin.unique', np.unique(img_bin))
    cv2.imwrite("Images/CNN1.png", img_bin*255)
    image_show_border("Images/CNN1.png", class_mat,
                      "Images/mumCNN1.png")
    x_idx_sec = sess.run(pred_idx, feed_dict={x_h: x_n_sec})
    x_pr_sec = sess.run(x_pred, feed_dict={x_h: x_n_sec})[:, 0].reshape(img_cut.shape)
    cv2.imwrite("x_pr_sec.png", x_pr_sec * 255)
    img_bin_sec, counter2 = vote_filter(x_idx_sec.reshape(img_cut.shape).reshape(img_cut.shape), window_size=1, thrshold=7)
    print('counter2:', counter2)
    # print('img_bin.unique', np.unique(img_bin))
    cv2.imwrite("Images/CNN" + str(img_num[test_num]) + ".png", img_bin_sec*255)
    image_show_border("Images/CNN" + str(img_num[test_num]) + ".png", class_mat_sec,
                      "Images/mumCNN" + str(img_num[test_num]) + ".png")

#w, W2, W3, b1, vis0 = BM_scratch(img_bin, x_pr,  class_mat, class_mat2, bm_epoch=15)


#x_pr_sec = x_pr_sec.reshape(img_cut.shape)
#M_predict(img_bin_sec, x_pr_sec, W1, W2, W3, b1, class_mat=class_mat_sec, class_mat_border=class_mat2_sec,
#           bm_epoch=15, error=.001, save_img=True)

