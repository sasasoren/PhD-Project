from spyder_window_maker import create_featureset_label, int_to_mat
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import random
from PIL import Image
import matplotlib
import datetime
import os


win_size = 1

n_nodes_hl1 = 7






#
# img = cv2.imread('org1-1.jpg', 0)
#
# mat_contents = sio.loadmat('lab_mat_wo')
# inside_m = mat_contents['labeled_mat']
# class_mat = inside_m[49:250, 49:250]
#
# img_cut = img[49:250, 49:250]


img = cv2.imread('org600.png',0)

mat_contents = sio.loadmat('lab_mat_wo_600')
y=mat_contents['labeled_mat']
y[y==4] = 1
class_mat = y[49:550 , 49:550]

img_cut = img[49:550 , 49:550]



train_x, train_y, test_x, test_y, x_, y_  = create_featureset_label(img, img_cut, class_mat)
print(train_x.shape)



b1 = np.load('a_b1.npy').astype('float')
w1 = np.load('a_w1.npy').astype('float')
b2 = np.load('b1.npy').astype('float')
w2 = np.load('w1.npy').astype('float')
b3 = np.load('b2.npy').astype('float')
w3 = np.load('w2.npy').astype('float')

hidden_1_layer = {'weight': tf.Variable(w1,name= 'w1'),
                  'bias': tf.Variable(b1,name= 'b1')}
#print(hidden_1_layer['weight'])
hidden_2_layer = {'weight': tf.Variable(w2,name= 'w1'),
                  'bias': tf.Variable(b2,name= 'b1')}

output_layer = {'weight': tf.Variable(w3,name='w2'),
                'bias': tf.Variable(b3,name='b2') }


def neural_network_model(data):
    with tf.name_scope('Hidden_1'):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)

    with tf.name_scope('Hidden_2'):
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

    with tf.name_scope('Output'):
        output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output


predict = neural_network_model(x_.astype('float'))
prob = tf.nn.softmax(neural_network_model(x_.astype('float')))

pred_idx = tf.argmax(predict, 1)


y_idx = tf.argmax(y_, 1)
correct = tf.equal(pred_idx , y_idx)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(predict, 1), num_classes=2)







with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())


    print('Accuracy:', accuracy.eval())

    print('Confusion:',confusion.eval())


    img_prob = prob[:,0].eval().reshape(img_cut.shape) *255
    img_bin = pred_idx.eval().reshape(img_cut.shape) * 255

    # cv2.imwrite("/home/sorena/Research/report/Images/img_prob.png", img_prob)
    # cv2.imwrite("/home/sorena/Research/report/Images/img_bin.png", img_bin)


img_bin[img_bin == 0] = 100
img_bin[img_bin == 255] = 0
img_bin[img_bin == 100] = 255







def window_square(img,img_cut,class_mat,win_size = 1, start_point = 49):
    windowset = []
    for i, j in np.ndindex(np.shape(img_cut)):
        # print(img_cut.shape)
        i1 = i + start_point
        j1 = j + start_point
        l = (2*win_size + 1)**2
        # print(i1,j1,i,j)
        windowset.append(list(np.concatenate([img[i1-win_size:i1+win_size+1,j1-win_size:j1+win_size+1].reshape(1,l).tolist()[0]
,int_to_mat(class_mat[i,j]),[class_mat[i,j]]])))

    # random.shuffle(windowset)
    return windowset


def ver_win(img, img_cut, class_mat, win_size=1, start_point=49):
    windowset = []
    for i1, j1 in np.ndindex(np.shape(img_cut)):
        # print(img_cut.shape)
        i = i1 + start_point
        j = j1 + start_point
        small_set = []
        for k in range(win_size + 1):
            if k == 0:
                small_set = list(img[i, j - (win_size - k):j + (win_size - k + 1)])
            else:
                small_set = small_set + list(img[i - k, j - (win_size - k):j + (win_size - k + 1)]) + list(
                    img[i + k, j - (win_size - k):j + (win_size - k + 1)])
                #  small_set.append(img[i-k,j-(win_size - k):j+(win_size - k+1)])
                #   small_set.append(list(img[i+k,j-(win_size - k):j+(win_size - k+1)]))

        features = list(small_set)
        classification = list(np.concatenate([int_to_mat(class_mat[i1, j1]), [class_mat[i1, j1]]]))

        windowset.append(features + classification)
    return windowset




def win_ftset_and_label(img, img_cut, class_mat, test_size=.1):
    s = int((img.shape[0]-img_cut.shape[0])/2)
    windowset = window_square(img, img_cut, class_mat,win_size = win_size,start_point= s)
    data = np.array(windowset)

    l = data.shape
    print(l)
    split = l[1] - 3
    testing_size = int(test_size * l[0])

    x = data[:, 0:split]
    y = data[:, split:(l[1]-1)]

    random.shuffle(windowset)
    data = np.array(windowset)

    train_x = data[:, 0:split][:-testing_size]
    train_y = data[:, split:(l[1]-1)][:-testing_size]
    test_x = data[:, 0:split][-testing_size:]
    test_y = data[:, split:(l[1]-1)][-testing_size:]

    label_x = data[:,l[1]-1][:-testing_size]
    return train_x, train_y, test_x, test_y, x, y , label_x

print(img_prob.shape)


d = img_prob.shape


train_x1, train_y1, test_x1, test_y1, x1_, y1_,l_x  = win_ftset_and_label(img_bin, img_bin[win_size:d[0]-win_size,win_size:d[1]-win_size], class_mat)
print(train_x1.shape)
print(train_x1[1])
print(train_y1[1])


mon_freq = 50
n_classes = 2
batch_size = 100
hm_epochs = 100
n_batch = len(train_x1)//batch_size+1


with tf.name_scope('Input_prob'):
    x = tf.placeholder('float')
    y = tf.placeholder('float')


hidden_1_layer_prob = {'weight': tf.Variable(tf.random_normal([len(train_x1[0]), n_nodes_hl1]),name= 'w1'),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]),name= 'b1')}
print(hidden_1_layer_prob['weight'].name)
output_layer_prob = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]),name='w2'),
                'bias': tf.Variable(tf.random_normal([n_classes]),name='b2') }
print(output_layer_prob['weight'].name)


def neural_network_model_prob(data,hidden_1 = hidden_1_layer_prob['weight'],hidden_1b =hidden_1_layer_prob['bias'], output_layer =output_layer_prob['weight'], output_layerb = output_layer_prob['bias']):
    with tf.name_scope('Hidden_1_prob'):
        l1 = tf.add(tf.matmul(data, hidden_1), hidden_1b)
        l1 = tf.nn.relu(l1)
        tf.summary.histogram("weight_1", hidden_1)

    with tf.name_scope('Output_prob'):
        output = tf.matmul(l1, output_layer) + output_layerb

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

iter_ = data_iterator(train_x1, train_y1, batch_size)





def train_neural_network(x):
    prediction = neural_network_model_prob(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    cost_sum = tf.summary.scalar("cost", cost)
    pred_idx = tf.argmax(prediction, 1)
    y_idx = tf.argmax(y, 1)
    correct = tf.equal(pred_idx , y_idx)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    accu_sum = tf.summary.scalar("Accuracy", accuracy)

    confusion = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(prediction, 1), num_classes=n_classes)

    # saver = tf.train.Saver()
    with tf.Session() as sess:
        summarymerged = tf.summary.merge_all()
        filename = "summary_log/prob-run"
        writer = tf.summary.FileWriter(filename, sess.graph)
        sess.run(tf.global_variables_initializer())

        j=0
        #epoch_loss = 0
        for epoch in range(hm_epochs):

            for i in range(n_batch):
                batch_x, batch_y = next(iter_)
                if i % mon_freq == 0:
                    j+=1
                    batch_loss , summ = sess.run([cost, summarymerged], feed_dict={x : batch_x, y : batch_y})
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

        # save_path = saver.save(sess, model_path)
        # print(save_path)
        print('Accuracy:', accuracy.eval({x: test_x1, y: test_y1}))

        print('Confusion:', sess.run(confusion, feed_dict={x: test_x1, y: test_y1}))
        global w1_np, b1_np, w2_np, b2_np
        w1_np = sess.run(tf.get_default_graph().get_tensor_by_name('w1_2:0'))
        b1_np = sess.run(tf.get_default_graph().get_tensor_by_name('b1_2:0'))

        w2_np = sess.run(tf.get_default_graph().get_tensor_by_name('w2_1:0'))
        b2_np = sess.run(tf.get_default_graph().get_tensor_by_name('b2_1:0'))
        print(b2_np.shape)


train_neural_network(x)



print(type(w1_np[0][0]))
print(type(b1_np[0]))
print(type(w2_np[0][0]))
print(type(b2_np[0]))
print(type(x1_.astype('float')[0][0]))
new_pred = neural_network_model_prob(x1_.astype('float32'),hidden_1 = w1_np ,hidden_1b = b1_np, output_layer = w2_np, output_layerb = b2_np)

new_pred_idx = tf.argmax(new_pred, 1)




pred_idx = tf.argmax(predict, 1)


new_y_idx = tf.argmax(y1_, 1)
correct = tf.equal(new_pred_idx , new_y_idx)
accuracy1 = tf.reduce_mean(tf.cast(correct, 'float'))

confusion1 = tf.confusion_matrix(labels=tf.argmax(y1_, 1), predictions=tf.argmax(new_pred, 1), num_classes=2)






with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())


    print('Accuracy:', accuracy1.eval())

    print('Confusion:',confusion1.eval())



    new_img_bin = new_pred_idx.eval().reshape(img_bin[win_size:d[0]-win_size,win_size:d[1]-win_size].shape) * 255
    print(new_img_bin[0])
    # cv2.imwrite("/home/sorena/Research/report/Images/squ_fil_img_bin.png", new_img_bin)







