import tensorflow as tf
import time
import random
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import sys

sys.path.insert(0, '../../dataloader')
from dataloader import DataLoader
from PIL import Image

num_batches = 50
batch_size = 20
epochs = 50

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# x_train = x_train[y_train[:,0]==1]
# print ("Training shape: {}".format(x_train.shape))

# x_train = (x_train - 127.5)/127.5


channels = 3


def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


def conv2d(inputs, kernel, filters, strides, padding):
    return tf.layers.conv2d(inputs, kernel_size=kernel, filters=filters, strides=strides, padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))


def conv2d_transpose(inputs, kernel, filters, strides, padding):
    return tf.layers.conv2d_transpose(inputs, kernel_size=kernel, filters=filters, strides=strides, padding=padding,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))


def leaky_on_batch_norm(inputs, is_training=True):
    return tf.nn.leaky_relu(tf.contrib.layers.batch_norm(inputs, is_training=is_training, epsilon=0.00001))


def dropout(inputs, keep_prob):
    return tf.nn.dropout(inputs, keep_prob)


"""
def generator(z,training, reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        #This is the generator model that is sepcifically designed to ouput 64x64 size images with the desired channels.
        hidden0= tf.layers.dense(z, 8*8*1024)
        hidden0 = leaky_on_batch_norm(hidden0)
        hidden0 = tf.reshape(hidden0, (-1, 8, 8, 1024))
        #hidden1=tf.layers.conv2d_transpose(inputs=z, kernel_size=[4,4], filters=1028*2, strides=(1, 1), padding='valid')
        #batch_norm1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden1, is_training=training, decay=momentum))
        hidden2=conv2d_transpose(hidden0, 4, 512, 2, 'same')
        batch_norm2 = leaky_on_batch_norm(hidden2)
        #batch_norm2 = dropout(batch_norm2, 0.5)
        hidden3 = conv2d_transpose(batch_norm2, 4, 256, 2, 'same')
        batch_norm3 = leaky_on_batch_norm(hidden3)
        hidden4=conv2d_transpose(batch_norm3, 4, 128, 2, 'same')
        batch_norm4 = leaky_on_batch_norm(hidden4)
        #batch_norm4 = dropout(batch_norm4, 0.5)
        hidden5=conv2d_transpose(batch_norm4, 4, 64, 1, 'same')
        batch_norm5 = leaky_on_batch_norm(hidden5)
        output= tf.nn.tanh(conv2d_transpose(batch_norm5, 4, channels, 1, 'same'))
        return output
        """


def generator(z, training, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        x = tf.layers.dense(z, 256 * 32 * 32)
        x = leaky_on_batch_norm(x)
        x = tf.reshape(x, (-1, 32, 32, 256))

        x = conv2d(x, 5, 256, 1, 'same')
        x = leaky_on_batch_norm(x)

        x = conv2d_transpose(x, 4, 256, 2, 'same')
        x = leaky_on_batch_norm(x)

        #adding attention
        # x = attention(x, 256)

        x = conv2d(x, 5, 256, 1, 'same')
        x = leaky_on_batch_norm(x)

        x = conv2d(x, 5, 256, 1, 'same')
        x = leaky_on_batch_norm(x)

        x = tf.nn.tanh(conv2d(x, 5, 3, 1, 'same'))
        return x


def discriminator(x, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        start_filters = 256
        hidden1 = conv2d(x, 3, start_filters, 1, 'same')
        #hidden1 = leaky_on_batch_norm(hidden1)
        hidden2 = conv2d(hidden1, 4, start_filters, 2, 'same')
        batch_norm2 = leaky_on_batch_norm(hidden2)
        #batch_norm2 = dropout(batch_norm2, 0.4)
        hidden3 = conv2d(batch_norm2, 4, start_filters, 2, 'same')
        batch_norm3 = leaky_on_batch_norm(hidden3)
        #batch_norm3 = dropout(batch_norm3, 0.5)
        hidden4 = conv2d(batch_norm3, 4, start_filters, 2, 'same')
        batch_norm4 = leaky_on_batch_norm(hidden4)
        logits = conv2d(batch_norm4, 4, start_filters, 2, 'same')
        logits = leaky_on_batch_norm(logits)
        logits = tf.layers.flatten(logits)
        logits = tf.nn.dropout(logits, 0.4)
        logits = tf.layers.dense(logits, 1)
        output=tf.sigmoid(logits)
        return output, logits

# def discriminator(x, reuse=None):
#     with tf.variable_scope('dis', reuse=reuse):
#         x = conv2d(x, 3, 256, 1, 'same')
#         x = leaky_on_batch_norm(x)
#
#         x = conv2d(x, 4, 256, 2, 'same')
#         x = leaky_on_batch_norm(x)
#
#         # adding attention
#         # x = attention(x, 256)
#
#         x = conv2d(x, 4, 256, 2, 'same')
#         x = leaky_on_batch_norm(x)
#
#         x = conv2d(x, 4, 256, 2, 'same')
#         x = leaky_on_batch_norm(x)
#
#         x = tf.layers.flatten(x)
#         x = tf.nn.dropout(x, 0.4)
#         logits = tf.layers.dense(x, 1)
#         output = tf.sigmoid(logits)
#         return output, logits

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


tf.reset_default_graph()

real_images = tf.placeholder(tf.float32, shape=[None, 64, 64, channels])
z = tf.placeholder(tf.float32, shape=[None, 100])
training = tf.placeholder(tf.bool)

noise_prop = 0.05

# true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
# flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
# true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

# gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
# flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
# gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]

# noisy_input_real = real_images + tf.random_normal(shape=tf.shape(real_images), mean=0.0, stddev=random.uniform(0.0, 0.1), dtype=tf.float32)

G = generator(z, training)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)

# tf.random_normal(shape=tf.shape(D_logits_real), mean=0.0, stddev=random.uniform(0.0, 0.1), dtype=tf.float32)

D_real_loss = loss_func(D_logits_real,
                        tf.zeros_like(D_logits_real) + tf.random_normal(shape=tf.shape(D_logits_real), mean=0.0,
                                                                        stddev=random.uniform(0.0, 0.1),
                                                                        dtype=tf.float32))
D_fake_loss = loss_func(D_logits_fake,
                        tf.ones_like(D_logits_fake) - tf.random_normal(shape=tf.shape(D_logits_real), mean=0.0,
                                                                       stddev=random.uniform(0.0, 0.1),
                                                                       dtype=tf.float32))

D_real_loss2 = loss_func(D_logits_real, tf.zeros_like(D_logits_real))
D_fake_loss2 = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

D_loss = (D_real_loss + D_fake_loss)
D_loss2 = D_real_loss2 + D_fake_loss2

G_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))

lr_g = 0.001
lr_d = 0.0001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

optimizer = tf.train.AdamOptimizer()

D_trainer = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(G_loss, var_list=g_vars)

D_trainer2 = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(D_loss2, var_list=d_vars)

D_gradients = 1
# optimizer.compute_gradients(D_loss, d_vars)
G_gradients = 1
# optimizer.compute_gradients(G_loss, g_vars)


init = tf.global_variables_initializer()

gen_samples = []

train_hist = {}
train_hist['D_losses_real'] = []
train_hist['D_losses_fake'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# load_files()

# rl_images = np.load("baklava.npy")
# rl_images = (rl_images - 127.5) / 127.5

d = DataLoader(mode='cat')

# print(x_train.shape)
# np.save('dataloader_test', d.get_batch_type(1, 31)[0])

with tf.Session() as sess:
    sess.run(init)
    print('Using our gen with max 1024 filters and smoothing labels and having noisy input')
    # train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    # train_set = (train_set - 0.5) * 2
    for epoch in range(epochs):
        epoch_start_time = time.time()
        D_losses_real = []
        D_losses_fake = []
        G_losses = []
        D_gradients = []
        G_gradients = []
        for i in range(num_batches):
            train_g = True
            train_d = True
            # batch_images = rl_images[i*batch_size:(i+1)*batch_size]
            batch_images = d.get_batch_type(batch_size, 11)[0]
            batch_images = np.reshape(batch_images, [-1, 64, 64, 3])
            batch_images = (batch_images - 127.5) / 127.5
            # batch_images = rl_images[i*batch_size:(i+1)*batch_size]

            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))

            loss_d_real, loss_d_fake, _ = sess.run([D_real_loss, D_fake_loss, D_trainer],
                                                   {real_images: batch_images, z: batch_z, training: True})
            D_losses_real.append(loss_d_real)
            D_losses_fake.append(loss_d_fake)
            loss_g_, _ = sess.run([G_loss, G_trainer], {z: batch_z, real_images: batch_images, training: True})
            # G_losses.append(loss_g_)
            # G_gradients.append(gradient_g)
            # if loss_d_ > loss_g_ * 2:
            #    train_g = False
            # if loss_g_ > loss_d_ * 2:
            #    train_d = False
            # if train_d:
            #    _ = sess.run([D_trainer], {real_images: batch_images, z: batch_z, training: True})
            # if epoch > 30 and loss_d_fake < 0.6:
            #    while loss_g_ > 1.0:
            #        loss_g_, _ = sess.run([G_loss, G_trainer], {real_images: batch_images, z: batch_z, training: True})
            if train_g:
                _ = sess.run([G_trainer], {real_images: batch_images, z: batch_z, training: True})
            # print('finished training batch')
            G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        sys.stdout.write('[%d/%d] - ptime: %.2f loss_d_real: %.3f, loss_d_fake: %.3f, loss_g: %.3f \n' % (
        (epoch + 1), epochs, per_epoch_ptime, np.mean(D_losses_real), np.mean(D_losses_fake), np.mean(G_losses)))
        sys.stdout.flush()
        train_hist['D_losses_real'].append(np.mean(D_losses_real))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        if epoch % 5 == 0 or epoch < 50:
            sample_z = np.random.uniform(-1, 1, size=(1, 100))
            gen_sample = sess.run(generator(z, training, reuse=True), feed_dict={z: sample_z, training: False})
            gen_samples.append(gen_sample)

# reshaped_rgb = gen_samples[epochs-1].reshape(32, 32, 3)
np.save('gen_samples_choc_cake_their_gen', gen_samples)
# img = Image.fromarray(reshaped_rgb, 'RGB')
# img.show()
# reshaped_rgb_last = gen_samples[epochs-1].reshape(64, 64, 3)
# np.save('reshaped_rgb_last_no_freeze3', reshaped_rgb_last)
# img_last = Image.fromarray(reshaped_rgb_last, 'RGB')
# img_last.show()

# plt.imshow(gen_samples[0].reshape(64, 64, 3))
# plt.show()
# plt.imshow(gen_samples[epochs-1].reshape(64, 64, 3))
# plt.show()

# plt.plot(train_hist['D_losses'])
# plt.plot(train_hist['G_losses'])
# plt.plot(train_hist['per_epoch_ptimes'])