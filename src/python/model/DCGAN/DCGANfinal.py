import tensorflow as tf
import time
import random
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.insert(0, '../../dataloader')
#from dataloader import get_batch, load_files
from PIL import Image

print('Hello World')

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = (np.concatenate((x_train, x_test), axis=0) - 127.5)/127.5


def show_imgs(batchidx):
  noise = np.random.normal(0, 1, size=(9, 1, 1, noise_dim))
  gen_imgs = generator.predict(noise)

  fig, axs = plt.subplots(3, 3)
  count = 0
  for i in range(3):
    for j in range(3):
      # Dont scale the images back, let keras handle it
      img = image.array_to_img(gen_imgs[count], scale=True)
      axs[i,j].imshow(img)
      axs[i,j].axis('off')
      count += 1
  plt.show()
  plt.close()


channels = 3

def generator(z,training, reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
            """ This is the generator model that is sepcifically designed to ouput 64x64 size images with the desired channels. """
            keep_prob=0.6
            momentum = 0.99
            hidden0=tf.layers.dense(z, 2*2*512)
            hidden0 = tf.reshape(hidden0, (-1, 2, 2, 512))
            hidden0 = tf.nn.leaky_relu(hidden0)
            #hidden1=tf.layers.conv2d_transpose(inputs=z, kernel_size=[4,4], filters=1028*2, strides=(1, 1), padding='valid')
            #batch_norm1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden1, is_training=training, decay=momentum))
            hidden2=tf.layers.conv2d_transpose(inputs=hidden0, kernel_size=5, filters=256, strides=(2, 2), padding='same')
            batch_norm2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden2, is_training=training, decay=momentum))
            hidden3=tf.layers.conv2d_transpose(inputs=batch_norm2, kernel_size=5, filters=128, strides=(2, 2), padding='same')
            batch_norm3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden3, is_training=training, decay=momentum))
            hidden4=tf.layers.conv2d_transpose(inputs=batch_norm3, kernel_size=5, filters=64, strides=(2, 2), padding='same')
            batch_norm4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden4, is_training=training, decay=momentum))
            output=tf.layers.conv2d_transpose(inputs=batch_norm4, kernel_size=5, filters=channels, strides=(2, 2), padding='same', activation=tf.nn.tanh)
            return output
def discriminator(X, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        momentum = 0.99
        #X = tf.reshape(X, shape=[-1, 64, 64, channels])
        #hidden0 = tf.layers.conv2d(inputs=X, kernel_size=5, filters=128, strides=2, padding='same', activation=tf.nn.leaky_relu)
       # batch_norm0 = tf.contrib.layers.batch_norm(hidden0)
        hidden1=tf.layers.conv2d(inputs=X, kernel_size=4, filters=64, strides=2, padding='same')
        #batch_norm1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden1, decay=momentum))
        hidden2=tf.layers.conv2d(inputs=hidden1, kernel_size=4, filters=128,strides=2, padding='same')
        batch_norm2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden2, decay=momentum))
        hidden3=tf.layers.conv2d(inputs=batch_norm2, kernel_size=4, filters=256,strides=2, padding='same')
        batch_norm3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(hidden3, decay=momentum))
        #x_flat = tf.contrib.layers.flatten(batch_norm3)
        logits=tf.layers.conv2d(inputs=batch_norm3, kernel_size=4, filters=1, strides=1, padding='valid')
        output=tf.sigmoid(logits)
        return output, logits

tf.reset_default_graph()

real_images=tf.placeholder(tf.float32,shape=[None, 32, 32, channels])
z=tf.placeholder(tf.float32,shape=[None, 1, 1, 100])
training=tf.placeholder(tf.bool)


#noisy_input_real = real_images + tf.random_normal(shape=tf.shape(real_images), mean=0.0, stddev=random.uniform(0.0, 0.1), dtype=tf.float32)

G=generator(z, training)
D_output_real,D_logits_real=discriminator(real_images)
D_output_fake,D_logits_fake=discriminator(G,reuse=True)

def loss_func(logits_in, labels_in):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss=loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))
D_loss = (D_real_loss + D_fake_loss)

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

lr_g = 0.0004
lr_d = 0.0002

tvars = tf.trainable_variables()
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(G_loss,var_list=g_vars)



batch_size=100
epochs=20
init=tf.global_variables_initializer()

gen_samples=[]


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


#load_files()

#rl_images = np.load("../../../../resources/processed/baklava.npy")
#rl_images = (rl_images - 127.5) / 127.5

print(x_train.shape)


with tf.Session() as sess:
    sess.run(init)
    print('Sess starting to run....')
    #train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    #train_set = (train_set - 0.5) * 2
    for epoch in range(epochs):
        epoch_start_time = time.time()
        D_losses=[]
        G_losses=[]
        print('starting epoch %d ...' % (epoch + 1))
        for i in range(x_train.shape[0]//batch_size):
            #print('we are now in %d' % (i))
            train_g=True
            train_d=True
            #batch_images = rl_images[i*batch_size:(i+1)*batch_size]
            batch_images = x_train[i*batch_size:(i+1)*batch_size]
            #np.save('image_test', batch_images[2])
            #print('just got batch')
            #batch_images = np.reshape(batch_images, [-1, 64, 64, 3])
            batch_z=np.random.uniform(-1, 1, size=(batch_size, 1, 1, 100))
            loss_d_, _ = sess.run([D_loss, D_trainer], {real_images: batch_images, z: batch_z, training: True})
            #print('just ran loss')
            D_losses.append(loss_d_)
            #z_ = np.random.normal(-1, 1, (batch_size, 1, 1, 100))
            loss_g_, _ = sess.run([G_loss, G_trainer], {z: batch_z, real_images: batch_images, training: True})
            #print('just ran gen loss')
            G_losses.append(loss_g_)
            #if loss_d_ > loss_g_ * 2:
             #   train_g = False
            #if loss_g_ > loss_d_ * 2:
             #   train_d = False
           # if train_d:
            #    _ = sess.run([D_trainer], {real_images: batch_images, z: batch_z, training: True})
            if train_g:
                _ = sess.run([G_trainer], {real_images: batch_images, z: batch_z, training: True})
            #print('finished training batch')
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
        sys.stdout.write('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f \n' % ((epoch + 1), epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
        sys.stdout.flush()
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)    
        sample_z=np.random.uniform(-1,1,size=(1, 1, 1, 100))
        gen_sample=sess.run(generator(z, training, reuse=True), feed_dict={z:sample_z, training: False})
        gen_samples.append(gen_sample)



reshaped_rgb = gen_samples[epochs-1].reshape(32, 32, 3)
np.save('gen_samples_CIFAR2', gen_samples)
img = Image.fromarray(reshaped_rgb, 'RGB')
img.show()
#reshaped_rgb_last = gen_samples[epochs-1].reshape(64, 64, 3)
#np.save('reshaped_rgb_last_no_freeze3', reshaped_rgb_last)
#img_last = Image.fromarray(reshaped_rgb_last, 'RGB')
#img_last.show()

# plt.imshow(gen_samples[0].reshape(64, 64, 3))
# plt.show()
# plt.imshow(gen_samples[epochs-1].reshape(64, 64, 3))
# plt.show()

#plt.plot(train_hist['D_losses'])
#plt.plot(train_hist['G_losses'])
#plt.plot(train_hist['per_epoch_ptimes'])
