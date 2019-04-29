import tensorflow as tf
import keras
import numpy as np
import matplotlib
import sys
import scipy
import random
#from ../dataloader.dataloader import get_batch
matplotlib.use('Agg')
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
'''newx = []
newy = []
for i in range(len(x_train)):
    if (y_train[i] == 5 or y_train[i] == 1):
        newy.append(y_train[i])
        newx.append(x_train[i])
x_train = newx
y_train = newy
'''

y_train = one_hot(y_train)
### GAN section

def generator(inp, y, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        bs = tf.shape(inp)[0]
        hidden1_im = tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden1_y = tf.layers.dense(inputs=y, units=2048, activation=tf.nn.leaky_relu)
        hidden2_y = tf.layers.dense(inputs=hidden1_y, units=1024, activation=tf.nn.leaky_relu)
        
        
        concat = tf.concat([hidden1_im, hidden2_y], 1)
        concat_dense = tf.layers.dense(inputs=concat, units=4*4*1024, activation = tf.nn.leaky_relu)
        preconv = tf.reshape(concat_dense, [bs,4, 4, 1024])


        #conv0a = tf.layers.conv2d_transpose(preconv, kernel_size=[5,5], filters=512, strides=(1,1),padding='valid')
        #conv0b = tf.layers.conv2d_transpose(preconv, kernel_size=[5,5], filters=512, strides=(1,1),padding='valid')
        conv1 = tf.layers.conv2d_transpose(preconv, kernel_size=[5,5], filters=1024, strides=(1,1),padding='valid')
        conv2 = tf.layers.conv2d_transpose(conv1, kernel_size=[5,5], filters=512, strides=(2,2), padding='same')
        conv3 = tf.layers.conv2d_transpose(conv2, kernel_size=[5,5], filters=256, strides=(2,2), padding='same')
        output = tf.layers.conv2d_transpose(conv3, kernel_size=[5,5], filters=1,strides=(2,2), padding='same')
        return output

# instead of putting y in discriminator, feed in vectorized real_images




def discriminator(img, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1_im = tf.layers.conv2d(img,  kernel_size=[5,5], filters=1024, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden1_pool = tf.layers.max_pooling2d(inputs=hidden1_im, pool_size=[2,2], strides=2)
        hidden2_im = tf.layers.conv2d(hidden1_pool,  kernel_size=[5,5], filters=512, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2_pool = tf.layers.max_pooling2d(inputs=hidden2_im, pool_size=[2,2], strides=2)
        hidden2_pool = tf.layers.flatten(hidden2_pool)
        output_im = tf.layers.dense(inputs=hidden2_pool, units=256, activation=tf.nn.leaky_relu)

        dense_0 = tf.layers.dense(inputs=output_im, units=256, activation=tf.nn.leaky_relu)
        
        dense_1f = tf.layers.dense(inputs=dense_0, units=128, activation=tf.nn.leaky_relu)
        dense_1c = tf.layers.dense(inputs=dense_0, units=128, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(dense_1f, units=1)
        output = tf.sigmoid(logits)

        classes_logits = tf.layers.dense(dense_1c, units=10)
        classes_output = tf.sigmoid(classes_logits)
        return logits, output, classes_logits #classes_output

real_images = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 1])
real_images += tf.random_normal(shape=tf.shape(real_images), mean=0.0, stddev=random.uniform(0.0,0.1),dtype=tf.float32) 
z = tf.placeholder(tf.float32, shape=[None, 100])
y1 = tf.placeholder(tf.float32, shape=[None, 10])
y2 = tf.placeholder(tf.float32, shape=[None, 10])
y3 = tf.placeholder(tf.float32, shape=[None, 10])
g = generator(z, y1)
dreallog, drealout, drealclasses = discriminator(real_images)
dfakelog, dfakeout, dfakeclasses = discriminator(g, reuse=True)

def loss_func(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def softmax_loss_func(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

drealloss = loss_func(dreallog, tf.ones_like(dreallog)*random.uniform(0.9,1))
dfakeloss = loss_func(dfakelog, tf.zeros_like(dfakelog))
drealclassloss = softmax_loss_func(drealclasses, y2)
dfakeclassloss = softmax_loss_func(dfakeclasses, y1)
dloss = (drealloss + dfakeloss)/4 + (drealclassloss + dfakeclassloss)/4

gloss = loss_func(dfakelog, tf.ones_like(dfakelog))/2 + dfakeclassloss/2

lrD = tf.placeholder(tf.float32, shape=[])
lrG = tf.placeholder(tf.float32, shape=[])

tvars=tf.trainable_variables()
dvars=[var for var in tvars if 'dis' in var.name]
gvars=[var for var in tvars if 'gen' in var.name]

dtrainer = tf.train.AdamOptimizer(lrD).minimize(dloss, var_list=dvars)
gtrainer = tf.train.AdamOptimizer(lrG).minimize(gloss, var_list=gvars)


epochs=100

init=tf.global_variables_initializer()
samples = []
lossds = []
lossgs = []
with tf.Session() as sess:
    sess.run(init)

    lrd = 0.00003
    lrg = 0.003
    for epoch in range(epochs):
        num_batches = len(y_train)//batch_size
        ld = 0
        lg = 0
        dcounter = 0
        dcounter = 0
        for i in range(num_batches):
            batch_im, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]#get_batch(batch_size)
            batch_im = batch_im

            batch_z=np.random.uniform(-1,1,size=(batch_size,100))
            d1 = sess.run(dloss, feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            g1 = sess.run(gloss, feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})
            drl = sess.run(drealloss,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})           
            dfl = sess.run(dfakeloss,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})           
            drcl = sess.run(drealclassloss,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})           
            dfcl = sess.run(dfakeclassloss,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y})           
            
            
            
            #if (g1 > 2*d1):
                #lrg = 0.001
                #lrd = 0.00001
            if (g1 < 0.9 and d1 < 0.9):
                lrd = 0.000006
                lrg = 0.0003
            #if (g1*2 < d1):
                #lrd = 0.0001
            ld += d1/num_batches
            lg += g1/num_batches
            print("Epoch ", epoch, "; batch #", i, "out of", num_batches, "genBatchLoss:", g1, "discBatchLoss:", d1, "lr Disc:", float(lrd), "lr Gen:", float(lrg), "discRealLoss:", drl, "discFakeLoss:", dfl, "discRealClassLoss:", drcl, "discFakeClassLoss:", dfcl)
            _=sess.run(dtrainer,feed_dict={real_images:batch_im,z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y,lrD:lrd})
            #if (epoch!=0 or i>300):
            print("running gen")

            _=sess.run(gtrainer,feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y,lrG:lrg})
            #if (g1 > d1*2):
            _=sess.run(gtrainer,feed_dict={z:batch_z,y1:batch_y,y2:batch_y,y3:batch_y,lrG:lrg})
            
        print("Finished Epoch", epoch)
        print("Generator Loss:", lg)
        print("Discriminator Loss:", ld)
        lossgs.append(lg)
        lossds.append(ld)
        oz = []
        for i in range(10):
            oz.append(i)
        #oz.append(1)
        oz = one_hot(oz)
        samplez=np.random.uniform(-1,1,size=(10,100))
        samples.append(sess.run(generator(z,y1,reuse=True), feed_dict={z:samplez,y1:oz}))
        np.save('ACGAN_data/samplesAll2', np.array(samples))
        np.save('ACGAN_data/discLossAll2', np.array(lossds))
        np.save('ACGAN_data/genLossAll2', np.array(lossgs))



plt.imshow(samples[0].reshape(64,64))
plt.show()
plt.imshow(samples[epochs-1].reshape(64,64))
plt.show()
