from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import datetime
import os

def neural_network_model(data, hidden_1_layer, output_layer):
    with tf.name_scope('Hidden_1'):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)
        tf.summary.histogram("weight_1", hidden_1_layer['weight'])

    with tf.name_scope('Output'):
        output = tf.matmul(l1, output_layer['weight']) + output_layer['bias']

    return output

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


def neural2(train_x, train_y, test_x, test_y, x_, hm_epochs=200, n_nodes_hl1=100, batch_size=100, n_classes=3,
                                other_test=None):

    path = os.getcwd()
    model_path = os.path.join(path, "model2_save", "model2.ckpt")



    mon_freq = 20000


    n_batch = len(train_x) // batch_size + 1

    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1]), name='w1'),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]), name='b1')}
    print(hidden_1_layer['weight'].name)
    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]), name='w2'),
                    'bias': tf.Variable(tf.random_normal([n_classes]), name='b2')}



    with tf.name_scope('Input'):
        x = tf.placeholder('float')
        y = tf.placeholder('float')

    iter_ = data_iterator(train_x, train_y, batch_size)

    prediction = neural_network_model(x, hidden_1_layer, output_layer)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    cost_sum = tf.summary.scalar("cost", cost)
    pred_idx = tf.argmax(prediction, 1)
    y_idx = tf.argmax(y, 1)
    correct = tf.equal(pred_idx , y_idx)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    accu_sum = tf.summary.scalar("Accuracy", accuracy)

    confusion = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(prediction, 1), num_classes=n_classes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        summarymerged = tf.summary.merge_all()
        filename = "summary_log/after-auto-run"
        writer = tf.summary.FileWriter(filename, sess.graph)
        sess.run(tf.global_variables_initializer())

        j=0
        #epoch_loss = 0
        for epoch in range(hm_epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter_)
                if i % mon_freq == 0:
                    j += 1
                    batch_loss, summ = sess.run([cost, summarymerged], feed_dict={x : batch_x, y : batch_y})
                    writer.add_summary(summ , j)
                    print("epoch = ", epoch, "iteration = ", i , "batch loss = ", batch_loss)
                #i = 0
                # while i < len(train_x):
                #     start = i
                #     end = i + batch_size
                #     batch_x = np.array(train_x[start:end])
                #     batch_y = np.array(train_y[start:end])tf.summary.scalar("loss",

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                #writer.add_summary(sumout)
                #epoch_loss = c
                #i += batch_size
            #epoch_loss , summ = sess.run([cost, summarymerged], feed_dict={x: batch_x,
                                                                 # y: batch_y})
            #tf.summary.scalar("loss", epoch_loss)
            #print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # w1_np = sess.run(tf.get_default_graph().get_tensor_by_name('w1:0'))
        # b1_np = sess.run(tf.get_default_graph().get_tensor_by_name('b1:0'))
        # np.save('w1.npy', w1_np)
        # np.save('b1.npy', b1_np)
        # w2_np = sess.run(tf.get_default_graph().get_tensor_by_name('w2:0'))
        # b2_np = sess.run(tf.get_default_graph().get_tensor_by_name('b2:0'))
        # np.save('w2.npy', w2_np)
        # np.save('b2.npy', b2_np)
        save_path = saver.save(sess, 'mlp_save/my_model.ckpt')
        print(save_path)



        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        acc = accuracy.eval({x: test_x, y: test_y})
        print('Confusion:',sess.run(confusion, feed_dict={x: test_x, y: test_y}))
        conf = sess.run(confusion, feed_dict={x: test_x, y: test_y})
        x_pred = sess.run(tf.nn.softmax(prediction), feed_dict={x: x_})
        pr_idx = sess.run(pred_idx, feed_dict={x: x_})
        print(x_pred.shape)
        if other_test.all() != None:
            pr_idx_other = sess.run(pred_idx, feed_dict={x: other_test})
            x_pred_other = sess.run(tf.nn.softmax(prediction), feed_dict={x: other_test})
            return np.array(x_pred), acc, conf, pr_idx, pr_idx_other, x_pred_other
        else:
            return np.array(x_pred), acc, conf, pr_idx


def main():

    train_x = np.load("Out_auto_x_train.npy")
    train_y = np.load("Out_auto_y_train.npy")
    test_x = np.load("Out_auto_x_test.npy")
    test_y = np.load("Out_auto_y_test.npy")
    x_ = train_x
    neural2(train_x, train_y, test_x, test_y, x_)
if __name__ == '__main__':
    tf.app.run(main)