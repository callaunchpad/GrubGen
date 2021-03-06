import tensorflow as tf
import keras
import numpy as np
import matplotlib
import sys
import scipy
import random
#from ../dataloader.dataloader import get_batch
matplotlib.use('TkAgg')
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims([scipy.misc.imresize(i, (64, 64, 1)) for i in x_train], axis=3)
x_test = np.expand_dims([scipy.misc.imresize(i, (64, 64, 1)) for i in x_test], axis=3)
batch_size=200
digits = [i for i in range(10)]
def one_hot(y_train):
    res = []
    for i in y_train:
        one = [0 for i in range(10)]
        one[i] = 1
        res += [one]
    return res

y_train = one_hot(y_train)[:60000]
### GAN section

def generator(inp, y, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        bs = tf.shape(inp)[0]
        hidden1_im = tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2_im = tf.layers.dense(inputs=hidden1_im, units=128, activation=tf.nn.leaky_relu)
        output_im = tf.layers.dense(inputs=hidden2_im, units=784, activation=tf.nn.leaky_relu)
        hidden1_y = tf.layers.dense(inputs=y, units=2048, activation=tf.nn.leaky_relu)
        hidden2_y = tf.layers.dense(inputs=hidden1_y, units=1024, activation=tf.nn.leaky_relu)
        hidden3_y = tf.layers.dense(inputs=hidden2_y, units=512, activation=tf.nn.leaky_relu)
        
        output_y = tf.layers.dense(inputs=hidden3_y, units=256, activation=tf.nn.leaky_relu)
        concat = tf.concat([output_im, output_y], 1)
        concat_dense = tf.layers.dense(inputs=concat, units=4*4*1024, activation = tf.nn.leaky_relu)
        preconv = tf.reshape(concat_dense, [bs,4, 4, 1024])



        conv1 = tf.layers.conv2d_transpose(preconv, kernel_size=[5,5], filters=2048, strides=(1,1),padding='valid')
        conv2 = tf.layers.conv2d_transpose(conv1, kernel_size=[5,5], filters=1024, strides=(1,1), padding='valid')
        conv3 = tf.layers.conv2d_transpose(conv2, kernel_size=[5,5], filters=512, strides=(1,1), padding='valid')
        
        conv4 = tf.layers.conv2d_transpose(conv3, kernel_size=[5,5], filters=256, strides=(2,2), padding='same')
        conv5 = tf.layers.conv2d_transpose(conv3, kernel_size=[5,5], filters=128, strides=(2,2), padding='same')
        output = tf.layers.conv2d_transpose(conv4, kernel_size=[5,5], filters=1,strides=(2,2), padding='same')
        return output

# instead of putting y in discriminator, feed in vectorized real_images




def discriminator(gen_inp, img_inp, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1_im = tf.layers.conv2d(gen_inp,  kernel_size=[5,5], filters=256, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden1_pool = tf.layers.max_pooling2d(inputs=hidden1_im, pool_size=[2,2], strides=2)
        hidden2_im = tf.layers.conv2d(hidden1_pool,  kernel_size=[5,5], filters=128, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2_pool = tf.layers.max_pooling2d(inputs=hidden2_im, pool_size=[2,2], strides=2)
        #flatten here
        hidden2_pool = tf.layers.flatten(hidden2_pool)
        output_im = tf.layers.dense(inputs=hidden2_pool, units=128, activation=tf.nn.leaky_relu)
        hidden1_y = tf.layers.dense(inputs=img_inp, units=256, activation=tf.nn.leaky_relu)
        hidden2_y = tf.layers.dense(inputs=hidden1_y, units = 256, activation = tf.nn.leaky_relu)
        hidden3_y = tf.layers.dense(inputs=hidden2_y, units = 256, activation = tf.nn.leaky_relu)
        
        output_y = tf.layers.dense(inputs=hidden3_y, units=128, activation=tf.nn.leaky_relu)

        dense_0 = tf.layers.dense(inputs=output_im, units=128, activation=tf.nn.leaky_relu)
        dense_1 = tf.layers.dense(inputs=dense_0, units=10, activation=tf.nn.leaky_relu)
        flatten_1 = tf.layers.flatten(dense_1)

        concat = tf.concat([flatten_1, output_y], 1)
        dense_2 = tf.layers.dense(inputs=concat, units=256, activation=tf.nn.leaky_relu)
        dense_3 = tf.layers.dense(inputs=dense_2, units=256, activation=tf.nn.leaky_relu)
        
        logits = tf.layers.dense(dense_3, units=1)
        output = tf.sigmoid(logits)
        return logits, output

real_images = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 1])
z = tf.placeholder(tf.float32, shape=[None, 100])
y1 = tf.placeholder(tf.float32, shape=[None, 10])
y2 = tf.placeholder(tf.float32, shape=[None, 10])
y3 = tf.placeholder(tf.float32, shape=[None, 10])
g = generator(z, y1)
dreallog, drealout = discriminator(real_images, y2)
dfakelog, dfakeout = discriminator(g, y3, reuse=True)

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

dtrainer = tf.train.AdamOptimizer(lr/50).minimize(dloss, var_list=dvars)
gtrainer = tf.train.AdamOptimizer(lr).minimize(gloss, var_list=gvars)


epochs=20

init=tf.global_variables_initializer()
samples = []
lossds = []
lossgs = []
with tf.Session() as sess:
    sess.run(init)
    print("STARTING")
    for epoch in range(epochs):
        num_batches = len(y_train)//batch_size
        ld = 0
        lg = 0
        dcounter = 0
        dcounter = 0
        for i in range(num_batches):
            
            batch_im, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]#get_batch(batch_size)

            
            batch_z=np.random.uniform(-1,1,size=(batch_size,100))
            d1= sess.run(dloss, feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            g1= sess.run(gloss, feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            tg = True
            td = True
            if (g1 > d1*2):
                td = False
            if (d1 > g1*2):
                tg = False
            print("Epoch ", epoch, "; batch #", i, "out of", num_batches, "discBatchLoss:",d1, "genBatchLoss:", g1, "Training Discriminator:", td, "Training Generator:", tg)
            if (td):
                _=sess.run(dtrainer,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            if (tg):
                _=sess.run(gtrainer,feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
                #_=sess.run(gtrainer,feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            ld += d1/num_batches
            lg += g1/num_batches
        print("Finished Epoch", epoch)
        print("Generator Loss:", lg)
        print("Discriminator Loss:", ld)
        lossgs.append(lg)
        lossds.append(ld)
        oz = []
        for i in range(10):
            oz.append(i)
        oz = one_hot(np.array(oz))
        samplez=np.random.uniform(-1,1,size=(10,100))
        samples.append(sess.run(generator(z,y1,reuse=True), feed_dict={z:samplez,y1:oz}))
        np.save('CGAN_data/samples3', np.array(samples))
        np.save('CGAN_data/discLoss3', np.array(lossds))
        np.save('CGAN_data/genLoss3', np.array(lossgs))



plt.imshow(samples[0].reshape(64,64))
plt.show()
plt.imshow(samples[epochs-1].reshape(64,64))
plt.show()
