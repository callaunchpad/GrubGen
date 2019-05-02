import tensorflow as tf
import numpy as np
import matplotlib
import sys
import scipy
import random
#from ../dataloader.dataloader import get_batch
matplotlib.use('Agg')
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

'''newx = []
newy = []
for i in range(len(x_train)):
    if (y_train[i] == 5 or y_train[i] == 1):
        newy.append(y_train[i])
        newx.append(x_train[i])
x_train = newx
y_train = newy
'''

class ACGAN_Model:
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=200, num_classes=10):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples = []
        self.lossds = []
        self.lossgs = []
        epochs=100
        self.model_init()


    def model_init(self):
        self.real_images = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
        self.real_images += tf.random_normal(shape=tf.shape(self.real_images), mean=0.0, stddev=random.uniform(0.0,0.1),dtype=tf.float32)
        self.z = tf.placeholder(tf.float32, shape=[None, 100])
        self.y1 = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.y2 = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        g = self.generator(self.z, self.y1)
        dreallog, _, drealclasses = self.discriminator(self.real_images)
        dfakelog, _, dfakeclasses = self.discriminator(g, reuse=True)

        self.drealloss = self.loss_func(dreallog, tf.ones_like(dreallog)*random.uniform(0.9,1))
        self.dfakeloss = self.loss_func(dfakelog, tf.zeros_like(dfakelog))
        self.drealclassloss = self.softmax_loss_func(drealclasses, self.y2)
        self.dfakeclassloss = self.softmax_loss_func(dfakeclasses, self.y1)
        self.dloss = (self.drealloss + self.dfakeloss)/4 + (self.drealclassloss + self.dfakeclassloss)/4

        self.gloss = self.loss_func(dfakelog, tf.ones_like(dfakelog))/2 + self.dfakeclassloss/2

        self.lrD = tf.placeholder(tf.float32, shape=[])
        self.lrG = tf.placeholder(tf.float32, shape=[])

        tvars=tf.trainable_variables()
        dvars=[var for var in tvars if 'dis' in var.name]
        gvars=[var for var in tvars if 'gen' in var.name]

        self.dtrainer = tf.train.AdamOptimizer(self.lrD).minimize(self.dloss, var_list=dvars)
        self.gtrainer = tf.train.AdamOptimizer(self.lrG).minimize(self.gloss, var_list=gvars)

    def loss_func(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    def softmax_loss_func(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    def generator(self, inp, y, reuse=None):
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
            output = tf.layers.conv2d_transpose(conv3, kernel_size=[5,5], filters=3,strides=(2,2), padding='same')
            return output

    def discriminator(self, img, reuse=None):
        with tf.variable_scope('dis',reuse=reuse):
            hidden1_im = tf.layers.conv2d(img,  kernel_size=[5,5], filters=1024, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
            hidden1_pool = tf.layers.max_pooling2d(inputs=hidden1_im, pool_size=[2,2], strides=2)
            hidden1_bn2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden1_pool))
            hidden2_im = tf.layers.conv2d(hidden1_bn2,  kernel_size=[5,5], filters=512, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
            hidden2_pool = tf.layers.max_pooling2d(inputs=hidden2_im, pool_size=[2,2], strides=2)
            hidden2_bn2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden2_pool))
            hidden3_im = tf.layers.conv2d(hidden2_pool,  kernel_size=[5,5], filters=512, strides=(2,2), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
            hidden3_pool = tf.layers.max_pooling2d(inputs=hidden3_im, pool_size=[2,2], strides=2)
            hidden3_pool = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden3_pool))
            hidden3_pool = tf.layers.flatten(hidden3_pool)
            output_im = tf.layers.dense(inputs=hidden3_pool, units=256, activation=tf.nn.leaky_relu)
            dense_0 = tf.layers.dense(inputs=output_im, units=256, activation=tf.nn.leaky_relu)

            dense_1f = tf.layers.dense(inputs=dense_0, units=128, activation=tf.nn.leaky_relu)
            dense_1c = tf.layers.dense(inputs=dense_0, units=128, activation=tf.nn.leaky_relu)
            logits = tf.layers.dense(dense_1f, units=1)
            output = tf.sigmoid(logits)

            classes_logits = tf.layers.dense(dense_1c, units=self.num_classes)
            return logits, output, classes_logits #classes_output

    def train(self,save_file, epochs=20, lrg=0.001, lrd=0.0001):
        init=tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(epochs):
                num_batches = len(self.y_train)//self.batch_size
                ld = 0
                lg = 0
                for i in range(num_batches):
                    batch_im, batch_y = self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size]#get_batch(batch_size)
                    batch_im = batch_im

                    batch_z=np.random.uniform(-1,1,size=(self.batch_size,100))
                    d1 = sess.run(self.dloss, feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y})
                    g1 = sess.run(self.gloss, feed_dict={self.z:batch_z,self.y1:batch_y,self.y2:batch_y})
                    drl = sess.run(self.drealloss,feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y})
                    dfl = sess.run(self.dfakeloss,feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y})
                    drcl = sess.run(self.drealclassloss,feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y})
                    dfcl = sess.run(self.dfakeclassloss,feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y})



                    #if (g1 > 2*d1):
                        #lrg = 0.001
                        #lrd = 0.00001
                    #if (g1 < 0.9 and d1 < 0.9):
                        #lrd = 0.000006
                        #lrg = 0.0003
                    #if (g1*2 < d1):
                        #lrd = 0.0001
                    ld += d1/num_batches
                    lg += g1/num_batches
                    print("Epoch ", epoch, "; batch #", i, "out of", num_batches, "genBatchLoss:", g1, "discBatchLoss:", d1, "lr Disc:", float(lrd), "lr Gen:", float(lrg), "discRealLoss:", drl, "discFakeLoss:", dfl, "discRealClassLoss:", drcl, "discFakeClassLoss:", dfcl)
                    _=sess.run(self.dtrainer,feed_dict={self.real_images:batch_im,self.z:batch_z,self.y1:batch_y,self.y2:batch_y,self.lrD:lrd})
                    #if (epoch!=0 or i>300):
                    print("running gen")
                    _=sess.run(self.gtrainer,feed_dict={self.z:batch_z,self.y1:batch_y,self.y2:batch_y,self.lrG:lrg})
                    #if (g1 > d1*2):
                    _=sess.run(self.gtrainer,feed_dict={self.z:batch_z,self.y1:batch_y,self.y2:batch_y,self.lrG:lrg})

                print("Finished Epoch", epoch)
                print("Generator Loss:", lg)
                print("Discriminator Loss:", ld)
                self.lossgs.append(lg)
                self.lossds.append(ld)
                oz = []
                for i in range(self.num_classes):
                    oz.append(i)
                #oz.append(1)
                oz = self.one_hot(oz)
                samplez=np.random.uniform(-1,1,size=(self.num_classes,100))
                self.samples.append(sess.run(self.generator(self.z, self.y1, reuse=True), feed_dict={self.z:samplez,self.y1:oz}))
                np.save('ACGAN_data/' + save_file, np.array(self.samples))
                np.save('ACGAN_data/' + save_file, np.array(self.lossds))
                np.save('ACGAN_data/' + save_file, np.array(self.lossgs))

    def generate(self, save_file):
        with tf.Session() as sess:
            oz = []
            for i in range(self.num_classes):
                oz.append(i)
            #oz.append(1)
            oz = self.one_hot(oz)
            samplez=np.random.uniform(-1,1,size=(self.num_classes,100))
            ret = sess.run(self.generator(self.z, self.y1, reuse=True), feed_dict={self.z:samplez,self.y1:oz})
            np.save('ACGAN_generate/' + save_file, np.array(ret))

    def one_hot(self, y):
        res = []
        for i in y:
            one = [0 for j in range(self.num_classes)]
            one[i] = 1
            res += [one]
        return res
