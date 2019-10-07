import numpy as np
import tensorflow as tf
a = np.arange(30).reshape(2,3,5)

print("a: ", a)

m1 = tf.constant([[3., 3.]])
b = np.amax(a, axis=0)
c = np.amax(a, axis=1)
d = np.amax(a, axis=2)
print("b: ", b)
print("c: ", c)

print("d: ", d)

with tf.Session() as sess:
    print(sess.run(m1))
    w = np.array([10,2])
    print(2*w)
