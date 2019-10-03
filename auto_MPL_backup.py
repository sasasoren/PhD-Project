"""Training first layer of deep learning by auto encoder"""
from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import datetime
import os

img = cv2.imread('org600.png', 0)

mat_contents = sio.loadmat('lab_mat_wo_600')
y = mat_contents['labeled_mat']
y[y == 4] = 1
class_mat = y[49:550, 49:550]

img_cut = img[49:550, 49:550]

train_x, train_y, test_x, test_y, _, _ = create_featureset_label(img, img_cut, class_mat)
print(train_x.shape)
n_nodes_hl1 = 8

path = os.getcwd()
model_path = os.path.join(path, "model_save", "model.ckpt")
print(model_path)
mon_freq = 50
n_classes = 13
batch_size = 100
hm_epochs = 100
n_batch = len(train_x) // batch_size + 1

# with tf.name_scope('Input'):
#     x = tf.placeholder('float')
#     y = tf.placeholder('float')

hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1]), name='a_w1'),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]), name='a_b1')}

# hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
# hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]), name='a_w2'),
                'bias': tf.Variable(tf.random_normal([n_classes]), name='a_b2')}
print(hidden_1_layer['weight'].name)


# def neural_network_model(data):
def nn_model(data):
    with tf.name_scope('Hidden_1'):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)
        # print("l1 : ",l1.shape)
        # tf.summary.histogram("weight_1", hidden_1_layer['weight'])
    #
    # with tf.name_scope('Hidden_2'):
    #     l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    #     l2 = tf.nn.relu(l2)
    #     tf.summary.histogram("weight_2", hidden_2_layer['weight'])
    #
    # with tf.name_scope('Hidden_3'):
    #     l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    #     l3 = tf.nn.relu(l3)
    #     tf.summary.histogram("weight_3", hidden_3_layer['weight'])
    #    print("out weight : ", output_layer['weight'])
    with tf.name_scope('Input_Output'):
        output = tf.matmul(l1, output_layer['weight']) + output_layer['bias']

    return output, l1


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


iter_ = data_iterator(train_x, train_y, batch_size)


# def train_neural_network(x):
def main(args):
    # prediction, _ = neural_network_model(x)

    with tf.name_scope('Input'):
        x = tf.placeholder('float')
        # y = tf.placeholder('float')
    prediction, _ = nn_model(x)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=prediction, labels=x))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    cost_sum = tf.summary.scalar("cost", cost)
    #
    # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summarymerged = tf.summary.merge_all()
        filename = "summary_log/auto-run"
        writer = tf.summary.FileWriter(filename, sess.graph)
        sess.run(tf.global_variables_initializer())

        j = 0
        # epoch_loss = 0
        for epoch in range(hm_epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter_)
                # print(batch_x.shape)
                if i % mon_freq == 0:
                    j += 1
                    batch_loss, summ = sess.run([cost, summarymerged], feed_dict={x: batch_x})
                    writer.add_summary(summ, j)
                    print("epoch = ", epoch, "iteration = ", i, "batch loss = ", batch_loss)
                # i = 0
                # while i < len(train_x):
                #     start = i
                #     end = i + batch_size
                #     batch_x = np.array(train_x[start:end])
                #     batch_y = np.array(train_y[start:end])tf.summary.scalar("loss",

                sess.run(optimizer, feed_dict={x: batch_x})
                # writer.add_summary(sumout)
                # epoch_loss = c
                # i += batch_size
                # epoch_loss , summ = sess.run([cost, summarymerged], feed_dict={x: batch_x,
                # y: batch_y})
                # tf.summary.scalar("loss", epoch_loss)
                # print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # a_w1_np = sess.run(tf.get_default_graph().get_tensor_by_name('a_w1:0'))
        # a_b1_np = sess.run(tf.get_default_graph().get_tensor_by_name('a_b1:0'))
        # np.save('a_w1.npy',a_w1_np)
        # np.save('a_b1.npy',a_b1_np)
        save_path = saver.save(sess, model_path)
        print(save_path)
        # print('Accuracy:', accuracy.eval({x: test_x}))


# train_neural_network(x)
def main():
    neural1(arg)


if __name__ == '__main__':
    tf.app.run(main)