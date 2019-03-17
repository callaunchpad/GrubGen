import tensorflow as tf 
import keras
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def generator(inp, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(inputs=hidden1, units=784, activation=tf.nn.tanh)
        return output

def discriminator(inp, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(hidden2, units=1)
        output = tf.sigmoid(logits)
        return logits, output

real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])
g = generator(z)
dreallog, drealout = discriminator(real_images)
dfakelog, dfakeout = discriminator(g,reuse=True)

def loss_func(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

drealloss = loss_func(dreallog, tf.ones_like(dreallog)*0.9)
dfakeloss = loss_func(dfakelog, tf.zeros_like(dfakelog))
dloss = drealloss + dfakeloss

gloss = loss_func(dfakelog, tf.ones_like(dfakelog))

lr = 0.001

tvars=tf.trainable_variables()
dvars=[var for var in tvars if 'dis' in var.name]
gvars=[var for var in tvars if 'gen' in var.name]

dtrainer = tf.train.AdamOptimizer(lr).minimize(dloss, var_list=dvars)
gtrainer = tf.train.AdamOptimizer(lr).minimize(gloss, var_list=gvars)

batch_size=100
epochs=200
init=tf.global_variables_initializer()
samples = []
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        num_batches = len(x_train)//batch_size
        for i in range(num_batches):
            batch_tr = x_train[i*batch_size:(i+1)*batch_size]
            batch_tr = np.reshape(batch_tr, (batch_size, 784))
            batch_z=np.random.uniform(-1,1,size=(batch_size,100))
            _=sess.run(dtrainer,feed_dict={real_images:batch_tr,z:batch_z})
            _=sess.run(gtrainer,feed_dict={z:batch_z})
        print("Epoch", epoch)

        samplez=np.random.uniform(-1,1,size=(1,100))
        samples.append(sess.run(generator(z,reuse=True), feed_dict={z:samplez}))
    
plt.imshow(samples[0].reshape(28,28))
plt.show()
plt.imshow(samples[epochs-1].reshape(28,28)) 
plt.show()
