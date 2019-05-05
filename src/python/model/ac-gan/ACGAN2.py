import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pickle as pkl
%matplotlib inline

sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

batch_size = 128
num_classes = 100

def discriminator(img):
    with tf.variable_scope('disc'):
        hidden1_im = tf.layers.conv2d(img,  kernel_size=[5,5], filters=64, strides=(1,1), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden2_im = tf.layers.conv2d(hidden1_im, kernel_size=[5,5], filters=64, strides=(1,1), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden3_im = tf.layers.conv2d(hidden2_im,  kernel_size=[5,5], filters=64, strides=(1,1), padding="SAME", activation=tf.nn.leaky_relu) #tf.layers.dense(inputs=inp, units=128, activation=tf.nn.leaky_relu)
        hidden3_pool = tf.layers.flatten(hidden3_im)
        output_im = tf.layers.dense(inputs=hidden3_pool, units=64, activation=tf.nn.leaky_relu)
        dense_0 = tf.layers.dense(inputs=output_im, units=64, activation=tf.nn.leaky_relu)
        dense_0 = tf.layers.dropout(inputs= dense_0)
        dense_1f = tf.layers.dense(inputs=dense_0, units=64, activation=tf.nn.leaky_relu)
        dense_1f = tf.layers.dropout(inputs= dense_1f)
        dense_1c = tf.layers.dense(inputs=dense_0, units=64, activation=tf.nn.leaky_relu)
        dense_1c = tf.layers.dropout(inputs= dense_1c)
        logits = tf.layers.dense(dense_1f, units=1)

        classes_logits = tf.layers.dense(dense_1c, units=self.num_classes)
        return logits, classes_logits 

def generator(inp, cond_vec):
        with tf.variable_scope('gen'):
            hidden1_im = tf.layers.dense(inputs=inp, units=64, activation=tf.nn.leaky_relu)
            hidden1_y = tf.layers.dense(inputs=y, units=1024, activation=tf.nn.leaky_relu)
            hidden2_y = tf.layers.dense(inputs=hidden1_y, units=512, activation=tf.nn.leaky_relu)


            concat = tf.concat([hidden1_im, hidden2_y], 1)
            concat_dense = tf.layers.dense(inputs=concat, units=4*4*1024, activation = tf.nn.leaky_relu)
            preconv = tf.reshape(concat_dense, [batch_size, 4, 4, 1024])


            conv0a = tf.layers.conv2d_transpose(preconv, kernel_size=[5,5], filters=2048, strides=(1,1),padding='valid', activation=tf.nn.leaky_relu)
            #conv0b = tf.layers.conv2d_transpose(conv0a, kernel_size=[5,5], filters=2048, strides=(1,1),padding='valid', activation=tf.nn.leaky_relu)
            conv1 = tf.layers.conv2d_transpose(conv0a, kernel_size=[5,5], filters=1024, strides=(2,2),padding='same', activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d_transpose(conv1, kernel_size=[5,5], filters=512, strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            #conv3 = tf.layers.conv2d_transpose(conv2, kernel_size=[5,5], filters=128, strides=(2,2), padding='same')
            output = tf.layers.conv2d_transpose(conv2, kernel_size=[5,5], filters=3,strides=(2,2), padding='same', activation='tanh')
            return output

def sigmoid_loss(logits, labels_in):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_in))

def softmax_loss(logits, labels_in):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_in))

def gan_loss(logits_real, logits_fake, class_real, class_fake, labels_in):
    class_real_loss = softmax_loss(class_real, labels_in)
    class_fake_loss = softmax_loss(class_fake, labels_in)

    gen_fake_loss = sigmoid_loss(logits_fake, tf.ones_like(logits_fake))
    disc_real_loss = sigmoid_loss(logits_real, tf.ones_like(logits_real))
    disc_fake_loss = sigmoid_loss(-logits_fake, tf.ones_like(logits_fake))
    
    return class_real_loss, class_fake_loss, gen_fake_loss, disc_real_loss, disc_fake_loss

real_images = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 3])
labels = tf.placeholder(tf.float32, shape=[None, num_classes])
z = tf.random_uniform((batch_size, batch_size)) * 2 - 1

G_sample = generator(z)
with tf.variable_scope("") as scope:
    logits_real, class_real = discriminator(real_images)
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    logits_fake, class_fake = discriminator(G_sample)

# get our loss
class_real_loss, class_fake_loss, gen_fake_loss, disc_real_loss, disc_fake_loss = gan_loss(logits_real, logits_fake, class_real, class_fake, labels)

G_loss = class_fake_loss + gen_fake_loss
D_loss = class_real_loss + disc_real_loss + disc_fake_loss

# setup training steps
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen') 
D_train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=D_vars)
G_train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'disc')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'gen')


init= tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

disc_loss_lst = []
gen_loss_lst = []

d = DataLoader("cat")

for i in range(10000):
    batch, batch_oh = d.get_batch_type(batch_size, 30)

    _, crl, drl, dfl = sess.run([D_train_step, class_real_loss, disc_real_loss, disc_fake_loss], feed_dict={
        real_images: batch, labels: batch_oh})

    _, cfl, gfl = sess.run([G_train_step, class_fake_loss, gen_fake_loss], feed_dict={
        real_images: batch, labels: batch_oh})

    lg = class_fake_loss + gen_fake_loss
    ld = class_real_loss + disc_real_loss + disc_fake_loss
    
    print("Iter", i, class_real_loss, class_fake_loss, disc_real_loss, disc_fake_loss, gen_fake_loss)

    disc_loss_lst.append(lg)
    gen_loss_lst.append(ld)
