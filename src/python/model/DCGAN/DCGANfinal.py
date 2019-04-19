import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.insert(0, '../../dataloader')
from dataloader import get_batch, load_files
from PIL import Image

print('Hello World')

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])



channels = 3

def generator(z, reuse=None):
	with tf.variable_scope('gen',reuse=reuse):
		""" This is the generator model that is sepcifically designed to ouput 64x64 size images with the desired channels. """


		keep_prob=0.6
		momentum = 0.99
		#is_training=True
		hidden1=tf.layers.conv2d_transpose(inputs=z, kernel_size=[4,4], filters=2048, strides=(1, 1), padding='valid', activation=tf.nn.leaky_relu)
		batch_norm1 = tf.contrib.layers.batch_norm(hidden1, decay=momentum)
		hidden2=tf.layers.conv2d_transpose(inputs=batch_norm1, kernel_size=[4,4], filters=1028, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		batch_norm2 = tf.contrib.layers.batch_norm(hidden2, decay=momentum)
		hidden3=tf.layers.conv2d_transpose(inputs=batch_norm2, kernel_size=[4,4], filters=512, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		batch_norm3 = tf.contrib.layers.batch_norm(hidden3, decay=momentum)
		hidden4=tf.layers.conv2d_transpose(inputs=batch_norm3, kernel_size=[4,4], filters=256, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		batch_norm4 = tf.contrib.layers.batch_norm(hidden4, decay=momentum)
		output=tf.layers.conv2d_transpose(inputs=batch_norm4, kernel_size=[4,4], filters=channels, strides=(2, 2), padding='same', activation=tf.nn.tanh)
		return output
def discriminator(X, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
    	momentum = 0.99
    	#X = tf.reshape(X, shape=[-1, 64, 64, channels])
    	hidden1=tf.layers.conv2d(inputs=X, kernel_size=4, filters=1028, strides=2, padding='same', activation=tf.nn.leaky_relu)
    	batch_norm1 = tf.contrib.layers.batch_norm(hidden1, decay=momentum)
    	hidden2=tf.layers.conv2d(inputs=batch_norm1, kernel_size=4, filters=512,strides=2, padding='same', activation=tf.nn.leaky_relu)
    	batch_norm2 = tf.contrib.layers.batch_norm(hidden2, decay=momentum)
    	hidden3=tf.layers.conv2d(inputs=batch_norm2, kernel_size=4, filters=256,strides=2, padding='same', activation=tf.nn.leaky_relu)
    	batch_norm3 = tf.contrib.layers.batch_norm(hidden3, decay=momentum)
    	#x_flat = tf.contrib.layers.flatten(batch_norm3)  	
    	logits=tf.layers.conv2d(inputs=batch_norm3, kernel_size=4, filters=1, strides=1, padding='valid')
    	output=tf.sigmoid(logits)
    	return output, logits

tf.reset_default_graph()

real_images=tf.placeholder(tf.float32,shape=[None, 64, 64, channels])
z=tf.placeholder(tf.float32,shape=[None, 1, 1, 100])

G=generator(z)
D_output_real,D_logits_real=discriminator(real_images)
D_output_fake,D_logits_fake=discriminator(G,reuse=True)

def loss_func(logits_in, labels_in):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss=loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))
D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

lr = 0.0002

tvars = tf.trainable_variables()
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(lr).minimize(G_loss,var_list=g_vars)


<<<<<<< HEAD

num_batches=30
=======
num_batches=40
>>>>>>> f8599a508a45ce40e67f47703a2eb45af084d550
batch_size=100
epochs=20
init=tf.global_variables_initializer()

gen_samples=[]


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


load_files()

with tf.Session() as sess:
    sess.run(init)
    print('Sess starting to run....')
    #train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    #train_set = (train_set - 0.5) * 2
    for epoch in range(epochs):
        epoch_start_time = time.time()
        D_losses=[]
        G_losses=[]
        print('starting epoch %d ...' % (epoch))
        for i in range(num_batches):
            #print('we are now in %d' % (i))
            train_g=True
            train_d=True
            batch_images = get_batch(batch_size)[0]
            #print('just got batch')
            batch_images = np.reshape(batch_images, [-1, 64, 64, 3])
            batch_z=np.random.uniform(-1, 1, size=(batch_size, 1, 1, 100))
            loss_d_ = sess.run([D_loss], {real_images: batch_images, z: batch_z})
            #print('just ran loss')
            D_losses.append(loss_d_)
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            loss_g_ = sess.run([G_loss], {z: batch_z, real_images: batch_images})
            #print('just ran gen loss')
            G_losses.append(loss_g_)
            if loss_d_ > loss_g_ * 2:
                train_g = False
            if loss_g_ > loss_d_ * 2:
                train_d = False
            if train_d:
                _ = sess.run([D_trainer], {real_images: batch_images, z: batch_z})
            if train_g:
                _ = sess.run([G_trainer], {real_images: batch_images, z: batch_z})
            #print('finished training batch')
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
        sys.stdout.write('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f \n' % ((epoch + 1), epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
        sys.stdout.flush()
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
            
        sample_z=np.random.uniform(-1,1,size=(1, 1, 1, 100))
        gen_sample=sess.run(generator(z, reuse=True), feed_dict={z:sample_z})
        gen_samples.append(gen_sample)



reshaped_rgb = gen_samples[0].reshape(64, 64, 3)
reshaped_rgb.astype('float32').tofile('reshaped_rgb_first2')
img = Image.fromarray(reshaped_rgb, 'RGB')
img.show()
reshaped_rgb_last = gen_samples[epochs-1].reshape(64, 64, 3)
reshaped_rgb_last.astype('float32').tofile('reshaped_rgb_last2')
img_last = Image.fromarray(reshaped_rgb_last, 'RGB')
img_last.show()

# plt.imshow(gen_samples[0].reshape(64, 64, 3))
# plt.show()
# plt.imshow(gen_samples[epochs-1].reshape(64, 64, 3))
# plt.show()

#plt.plot(train_hist['D_losses'])
#plt.plot(train_hist['G_losses'])
#plt.plot(train_hist['per_epoch_ptimes'])
