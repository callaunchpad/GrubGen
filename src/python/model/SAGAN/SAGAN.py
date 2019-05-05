import tensorflow as tf
import time
import numpy as np
import os
import scipy
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import sys

sys.path.insert(0, "../../dataloader")
# print(sys.path)
# import os

from dataloader import DataLoader
from PIL import Image

channels = 1


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])




# dim of z is [batch, 1, 1, 100]
def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        """ This is the generator model that is specifically designed to ouput 64x64 size images with the desired channels. """
        # dim [batch size, 1, 1, 100]
        keep_prob = 0.6
        momentum = 0.99
        # is_training=True
        hidden0 = tf.layers.dense(z, 32 * 32 * 128)
        hidden0 = tf.nn.leaky_relu(hidden0)
        hidden0 = tf.reshape(hidden0, (-1, 32, 32, 128))

        hidden2 = tf.layers.conv2d_transpose(inputs=hidden0, kernel_size=[4, 4], filters=128, strides=2, padding='same')
        batch_norm2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden2, is_training=True))

        hidden3 = tf.layers.conv2d(inputs=batch_norm2, kernel_size=[4, 4], filters=128, strides=(1, 1), padding='same')
        batch_norm3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden3, decay=momentum))
        # batch_norm3_attention = attention(batch_norm3, 128)

        hidden4 = tf.layers.conv2d(inputs=batch_norm3, kernel_size=[4, 4], filters=128, strides=(1, 1), padding='same')
        batch_norm4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden4, decay=momentum))

        # batch size, 32, 32, 128
        output = tf.layers.conv2d(inputs=batch_norm4, kernel_size=[4, 4], filters=channels, strides=(1, 1), padding='same')
        output = tf.nn.tanh(output)

        # batch size, 64, 64, 3
        return output

def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):

        hidden1 = tf.layers.conv2d(inputs=X, kernel_size=4, filters=512, strides=2, padding='same')
        batch_norm1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden1))
        # batch size, 32, 32, 128
        hidden2 = tf.layers.conv2d(inputs=batch_norm1, kernel_size=4, filters=512, strides=2, padding='same')
        batch_norm2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden2))
        # batch size, 16, 16, 256

        # batch_norm2_attention = attention(batch_norm2, 512)

        hidden3 = tf.layers.conv2d(inputs=batch_norm2, kernel_size=4, filters=512, strides=2, padding='same')
        batch_norm3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden3))
        # batch size, 8, 8, 512

        logits = tf.layers.conv2d(inputs=batch_norm3, kernel_size=4, filters=1, strides=1, padding='valid')
        logits = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(logits))

        logits = tf.contrib.layers.flatten(logits)
        logits = tf.nn.dropout(logits, 0.4)
        logits = tf.layers.dense(logits, 1)
        # batch size, ?, ?, 1
        output = tf.sigmoid(logits)
        return output, logits


def attention(x, channels):
    with tf.variable_scope('attention', reuse=None):
        f = tf.layers.conv2d(inputs=x, kernel_size=1, filters=channels // 8,
                             strides=1)  # bs, h, w, filters (channels/8)
        g = tf.layers.conv2d(inputs=x, kernel_size=1, filters=channels // 8,
                             strides=1)  # bs, h, w, filters (channels/8)
        h = tf.layers.conv2d(inputs=x, kernel_size=1, filters=channels, strides=1)  # bs, h, w, filters (channels)

        reshape_f = tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]])  # bs, h*w, filters (channels/8)
        reshape_g = tf.reshape(g, shape=[g.shape[0], -1, g.shape[-1]])  # bs, h*w, filters (channels/8)
        reshape_h = tf.reshape(h, shape=[h.shape[0], -1, h.shape[-1]])  # bs, h*w, filters (channels)

        s = tf.matmul(reshape_g, reshape_f, transpose_b=True)  # bs, h*w, h*w
        beta = tf.nn.softmax(s)  # the attention map
        o = tf.matmul(beta, reshape_h)  # bs, h*w, filters (channels)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.shape)  # bs, h, w, filters (channels)
        x = gamma * o + x
    return x

# def spectral_norm(w, iteration=1):
#     w_shape = w.shape.as_list()
#     w = tf.reshape(w, [-1, w_shape[-1]])
#
#     u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
#
#     u_hat = u
#     v_hat = None
#     for i in range(iteration):
#         """
#         power iteration
#         Usually iteration = 1 will be enough
#         """
#         v_ = tf.matmul(u_hat, tf.transpose(w))
#         v_hat = l2_norm(v_)
#
#         u_ = tf.matmul(v_hat, w)
#         u_hat = l2_norm(u_)
#
#     sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
#     w_norm = w / sigma
#
#     with tf.control_dependencies([u.assign(u_hat)]):
#         w_norm = tf.reshape(w_norm, w_shape)
#
#     return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


tf.reset_default_graph()

num_batches = 10
batch_size = 50
epochs = 15

real_images = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, channels], name='real_images')
z = tf.placeholder(tf.float32, shape=[batch_size, 1, 1, 100], name='noise')

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)


def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_logits_real,
                        tf.zeros_like(D_logits_real) + tf.random_normal(shape=tf.shape(D_logits_real), mean=0.0,
                                                                        stddev=np.random.uniform(0.0, 0.1),
                                                                        dtype=tf.float32))
D_fake_loss = loss_func(D_logits_fake,
                        tf.ones_like(D_logits_fake) - tf.random_normal(shape=tf.shape(D_logits_fake), mean=0.0,
                                                                       stddev=np.random.uniform(0.0, 0.1),
                                                                       dtype=tf.float32))
D_loss = (D_real_loss + D_fake_loss)

G_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))

lr_d = 0.004
lr_g = 0.001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(lr_d).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(lr_g).minimize(G_loss, var_list=g_vars)

init = tf.global_variables_initializer()

gen_samples = []

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# d = DataLoader(mode="cat")

with tf.Session() as sess:
    sess.run(init)
    train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    train_set = (train_set - 0.5) * 2
    for epoch in range(epochs):
        epoch_start_time = time.time()
        D_losses = []
        G_losses = []
        for i in range(num_batches):
            print(train_set.shape)
            print(i)
            train_g = True
            train_d = True
            batch_images = train_set[i*batch_size:(i+1)*batch_size]
            # batch_images = d.get_batch_type(batch_size, 61)[0]
            # batch_images = np.reshape(batch_images, [-1, 64, 64, 3])
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 1, 1, 100))

            loss_d_, _ = sess.run([D_loss, D_trainer], {real_images: batch_images, z: batch_z})
            D_losses.append(loss_d_)
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            loss_g_, _ = sess.run([G_loss, G_trainer], {z: batch_z, real_images: batch_images})
            G_losses.append(loss_g_)
            # if loss_d_ > loss_g_ * 2:
            #     train_g = False
            # if loss_g_ > loss_d_ * 2:
            #     train_d = False
            # if train_d:
            #     _ = sess.run([D_trainer], {real_images: train_set, z: batch_z})
            # if train_g:
            #     _ = sess.run([G_trainer], {real_images: train_set, z: batch_z})
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        sys.stdout.write('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f \n' % (
        (epoch + 1), epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
        sys.stdout.flush()
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        sample_z = np.random.uniform(-1, 1, size=(batch_size, 1, 1, 100))

        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})

        gen_samples.append(gen_sample)

# reshaped_rgb = gen_samples[0].reshape(64, 64, 3)
# img = Image.fromarray(reshaped_rgb, 'RGB')
# img.show()

# reshaped_rgb_last = gen_samples[epochs-1].reshape(64, 64, 3)
# img_last = Image.fromarray(reshaped_rgb_last, 'RGB')
# img_last.show()
gen_samples = np.array(gen_samples)

# gen_samples shape = [epochs, 1, 64, 64, 3]
np.save('first.npy', gen_samples[0])
np.save('last.npy', gen_samples[epochs - 1])

# plt.imshow(gen_samples[0].reshape(64, 64, 3))
# plt.show()
# plt.imshow(gen_samples[epochs-1].reshape(64, 64, 3))
# plt.show()

# plt.plot(train_hist['D_losses'])
# plt.plot(train_hist['G_losses'])
# plt.plot(train_hist['per_epoch_ptimes'])
