"""Training first layer of deep learning by auto encoder"""
from spyder_window_maker import create_featureset_label
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import datetime
import os


img = cv2.imread('org1-1.jpg',0)

mat_contents = sio.loadmat('lab_mat_wo')
y=mat_contents['labeled_mat']
y[y==4] = 1
class_mat = y[49:250, 49:250]

img_cut = img[49:250, 49:250]

train_x, train_y, test_x, test_y,x_, y_ = create_featureset_label(img, img_cut, class_mat)
print(train_x.shape)
n_nodes_hl1 = 8

n_nodes_hl2 = 100

path = os.getcwd()
model_path = os.path.join(path, "model_save","model.ckpt")
print(model_path)
mon_freq = 50
n_classes = 13
n_classes2 = 2
batch_size = 100
hm_epochs = 100
n_batch = len(train_x)//batch_size+1

# with tf.name_scope('Input'):
#     x = tf.placeholder('float')
#     y = tf.placeholder('float')

hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1]),name='a_w1'),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]),name='a_b1')}
#
# hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
# hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]),name='a_w2'),
                'bias': tf.Variable(tf.random_normal([n_classes]),name='a_b2') }


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

x = tf.placeholder('float')
# def train_neural_network(x):
def Firt_Autoencoder(x):
    # prediction, _ = neural_network_model(x)

    # with tf.name_scope('Input'):
        # x = tf.placeholder('float')
        # y = tf.placeholder('float')
    prediction, l1 = nn_model(x)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions =prediction, labels=x))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    cost_sum = tf.summary.scalar("cost", cost)
    #
    # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summarymerged = tf.summary.merge_all()
        filename = "summary_log/whole"
        writer = tf.summary.FileWriter(filename, sess.graph)
        sess.run(tf.global_variables_initializer())

        j=0
        #epoch_loss = 0
        for epoch in range(hm_epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter_)
                # print(batch_x.shape)
                if i % mon_freq == 0:
                    j+=1
                    batch_loss , summ = sess.run([cost, summarymerged], feed_dict={x : batch_x})
                    writer.add_summary(summ , j)
                    print("epoch = ", epoch, "iteration = ", i , "batch loss = ", batch_loss)
                #i = 0
                # while i < len(train_x):
                #     start = i
                #     end = i + batch_size
                #     batch_x = np.array(train_x[start:end])
                #     batch_y = np.array(train_y[start:end])tf.summary.scalar("loss",

                sess.run(optimizer, feed_dict={x: batch_x})
                #writer.add_summary(sumout)
                #epoch_loss = c
                #i += batch_size
            #epoch_loss , summ = sess.run([cost, summarymerged], feed_dict={x: batch_x,
                                                                 # y: batch_y})
            #tf.summary.scalar("loss", epoch_loss)
            #print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)



        # a_w1_np = sess.run(tf.get_default_graph().get_tensor_by_name('a_w1:0'))
        # a_b1_np = sess.run(tf.get_default_graph().get_tensor_by_name('a_b1:0'))
        # np.save('a_w1.npy',a_w1_np)
        # np.save('a_b1.npy',a_b1_np)
        global l1_eval_train, l1_eval_test, l1_eval_x
        l1_eval_train = sess.run(l1, feed_dict={x: train_x})
        print(l1_eval_train.shape)
        l1_eval_test = sess.run(l1, feed_dict={x: test_x})
        l1_eval_x = sess.run(l1, feed_dict={x: x_})
        save_path = saver.save(sess, model_path)
        print(save_path)
        # print('Accuracy:', accuracy.eval({x: test_x}))
# train_neural_network(x)
# if __name__ == '__main__':
#     tf.app.run(main)

Firt_Autoencoder(x)







hidden_2_layer = {'weight': tf.Variable(tf.random_normal([len(l1_eval_train[0]), n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer2 = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes2])),
                'bias': tf.Variable(tf.random_normal([n_classes2])) }



def neural_network_model(data):
    with tf.name_scope('Hidden_2'):
        l2 = tf.add(tf.matmul(data, hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)
        tf.summary.histogram("weight_2", hidden_2_layer['weight'])

    with tf.name_scope('Output2'):
        output = tf.matmul(l2, output_layer2['weight']) + output_layer2['bias']

    return output


iter2_ = data_iterator(l1_eval_train, train_y, batch_size)

x_in = tf.placeholder('float')
y = tf.placeholder('float')


def train_neural_network(x_in):
    prediction = neural_network_model(x_in)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    cost_sum = tf.summary.scalar("cost2", cost)
    pred_idx = tf.argmax(prediction, 1)
    y_idx = tf.argmax(y, 1)
    correct = tf.equal(pred_idx , y_idx)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    accu_sum = tf.summary.scalar("Accuracy2", accuracy)

    confusion = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(prediction, 1), num_classes=n_classes2)

    # saver = tf.train.Saver()
    with tf.Session() as sess1:
        summarymerged = tf.summary.merge_all()
        filename = "summary_log/whole"
        writer = tf.summary.FileWriter(filename, sess1.graph)
        sess1.run(tf.global_variables_initializer())

        j=0
        #epoch_loss = 0
        for epoch in range(hm_epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter2_)
                if i % mon_freq == 0:
                    j+=1
                    batch_loss , summ = sess1.run([cost, summarymerged], feed_dict={x_in : batch_x, y : batch_y})
                    writer.add_summary(summ , j)
                    print("epoch = ", epoch, "iteration = ", i , "batch loss = ", batch_loss)
                #i = 0
                # while i < len(train_x):
                #     start = i
                #     end = i + batch_size
                #     batch_x = np.array(train_x[start:end])
                #     batch_y = np.array(train_y[start:end])tf.summary.scalar("loss",

                sess1.run(optimizer, feed_dict={x_in: batch_x, y: batch_y})
                #writer.add_summary(sumout)
                #epoch_loss = c
                #i += batch_size
            #epoch_loss , summ = sess1.run([cost, summarymerged], feed_dict={x: batch_x,
                                                                 # y: batch_y})
            #tf.summary.scalar("loss", epoch_loss)
            #print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # w1_np = sess1.run(tf.get_default_graph().get_tensor_by_name('w1:0'))
        # b1_np = sess1.run(tf.get_default_graph().get_tensor_by_name('b1:0'))
        # np.save('w1.npy', w1_np)
        # np.save('b1.npy', b1_np)
        # w2_np = sess1.run(tf.get_default_graph().get_tensor_by_name('w2:0'))
        # b2_np = sess1.run(tf.get_default_graph().get_tensor_by_name('b2:0'))
        # np.save('w2.npy', w2_np)
        # np.save('b2.npy', b2_np)
        # save_path = saver.save(sess1, model_path)
        # print(save_path)


        print('Accuracy2:', accuracy.eval({x_in: l1_eval_test, y: test_y}))

        print('Confusion2:',sess1.run(confusion, feed_dict={x_in: l1_eval_test, y: test_y}))


train_neural_network(x_in)


predict = neural_network_model(l1_eval_x.astype('float'))

pred_idx = tf.argmax(predict, 1)

with tf.Session() as sess2:

    sess2.run(tf.global_variables_initializer())

    img_bin = pred_idx.eval().reshape(img_cut.shape) * 255

    cv2.imwrite("/home/sorena/Research/report/Images/img_bin600_whole.png", img_bin)
