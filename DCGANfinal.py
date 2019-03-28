import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])


def generator(z, reuse=None):
	with tf.variable_scope('gen',reuse=reuse):
		keep_prob=0.6
		momentum = 0.99
		# is_training=True
		# activation=tf.nn.leaky_relu
		# x = z
		# d1 = 4
		# d2 = 1
		# x = tf.layers.dense(x, units=4*4*1, activation=activation)
		# x = tf.layers.dropout(x, keep_prob)
		# x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
		# x = tf.reshape(x, shape=[-1, d1, d1, d2])
		# x = tf.image.resize_images(x, size=[7, 7])
		# x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
		# x = tf.layers.dropout(x, keep_prob)
		# x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
		# x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
		# x = tf.layers.dropout(x, keep_prob)
		# x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
		# x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
		# x = tf.layers.dropout(x, keep_prob)
		# x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
		# x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
		# return x
		#hidden=tf.layers.dense(inputs=z,units=4*4*1,activation=tf.nn.leaky_relu)
		#dropout_=tf.layers.dropout(inputs=hidden1, rate=keep_prob)
		#batch_norm = tf.contrib.layers.batch_norm(hidden, decay=0.9)
		#reshape1 = tf.reshape(batch_norm, shape=[-1, 4, 4, 1])
		#full_reshape = tf.image.resize_images(reshape1, size=[7, 7])
		hidden1=tf.layers.conv2d_transpose(inputs=z, kernel_size=[4,4], filters=1028, strides=(1, 1), padding='valid', activation=tf.nn.leaky_relu)
		#dropout_1=tf.layers.dropout(hidden1, rate=keep_prob)
		batch_norm1 = tf.contrib.layers.batch_norm(hidden1, decay=momentum)
		hidden2=tf.layers.conv2d_transpose(inputs=batch_norm1, kernel_size=[4,4], filters=512, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		#dropout_2=tf.layers.dropout(hidden2, rate=keep_prob)
		batch_norm2 = tf.contrib.layers.batch_norm(hidden2, decay=momentum)
		hidden3=tf.layers.conv2d_transpose(inputs=batch_norm2, kernel_size=[4,4], filters=256, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		#dropout_3=tf.layers.dropout(hidden3, rate=keep_prob)
		batch_norm3 = tf.contrib.layers.batch_norm(hidden3, decay=momentum)
		hidden4=tf.layers.conv2d_transpose(inputs=batch_norm3, kernel_size=[4,4], filters=128, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
		#dropout_4=tf.layers.dropout(hidden4, rate=keep_prob)
		batch_norm4 = tf.contrib.layers.batch_norm(hidden4, decay=momentum)
		output=tf.layers.conv2d_transpose(inputs=batch_norm4, kernel_size=[4,4], filters=1, strides=(2, 2), padding='same', activation=tf.nn.tanh)
		return output
def discriminator(X, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
    	momentum = 0.9
    	#x = tf.reshape(X, shape=[-1, 28, 28, 1])
    	hidden1=tf.layers.conv2d(inputs=X, kernel_size=3, filters=128, strides=2, padding='same', activation=tf.nn.leaky_relu)
    	#dropout_1= tf.layers.dropout(inputs=hidden1, rate=0.2)
    	batch_norm1 = tf.contrib.layers.batch_norm(hidden1, decay=momentum)
    	hidden2=tf.layers.conv2d(inputs=batch_norm1, kernel_size=3, filters=256,strides=2, padding='same', activation=tf.nn.leaky_relu)
    	#dropout_2= tf.layers.dropout(inputs=hidden2, rate=0.2)
    	batch_norm2 = tf.contrib.layers.batch_norm(hidden2, decay=momentum)
    	hidden3=tf.layers.conv2d(inputs=batch_norm2, kernel_size=3, filters=512,strides=2, padding='same', activation=tf.nn.leaky_relu)
    	#dropout_3= tf.layers.dropout(inputs=hidden3, rate=0.2)
    	batch_norm3 = tf.contrib.layers.batch_norm(hidden3, decay=momentum)
    	# x_flat = tf.contrib.layers.flatten(batch_norm3)
    	# pre_output=tf.layers.dense(x_flat, units=1, activation=tf.nn.leaky_relu)
    	#hidden4=tf.layers.conv2d(inputs=batch_norm3, kernel_size=4, filters=512,strides=2, padding='same', activation=tf.nn.leaky_relu)
    	#batch_norm4 = tf.contrib.layers.batch_norm(hidden4, decay=momentum)    	
    	logits=tf.layers.conv2d(inputs=batch_norm3, kernel_size=4, filters=1, strides=1, padding='valid')
    	output=tf.sigmoid(logits)
    	return output, logits

tf.reset_default_graph()

real_images=tf.placeholder(tf.float32,shape=[None, 64, 64, 1])
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

lr = 0.002

tvars = tf.trainable_variables()
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(lr).minimize(G_loss,var_list=g_vars)

batch_size=100
epochs=20
init=tf.global_variables_initializer()

gen_samples=[]


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []



with tf.Session() as sess:
	sess.run(init)
	train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
	train_set = (train_set - 0.5) * 2
	for epoch in range(epochs):
		epoch_start_time = time.time()
		D_losses=[]
		G_losses=[]
		num_batches=mnist.train.num_examples//batch_size
		for i in range(num_batches):
			train_g=True
			train_d=True
			batch_images = train_set[i*batch_size:(i+1)*batch_size]
			batch_z=np.random.uniform(-1, 1, size=(batch_size, 1, 1, 100))
			loss_d_ = sess.run([D_loss], {real_images: batch_images, z: batch_z})
			D_losses.append(loss_d_)
			z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
			loss_g_ = sess.run([G_loss], {z: batch_z, real_images: batch_images})
			G_losses.append(loss_g_)
			if loss_d_ > loss_g_ * 2:
				train_g = False
			if loss_g_ > loss_d_ * 2:
				train_d = False
			if train_d:
				_ = sess.run([D_trainer], {real_images: batch_images, z: batch_z})
			if train_g:
				_ = sess.run([G_trainer], {real_images: batch_images, z: batch_z})
		epoch_end_time = time.time()
		per_epoch_ptime = epoch_end_time - epoch_start_time
		print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
		train_hist['D_losses'].append(np.mean(D_losses))
		train_hist['G_losses'].append(np.mean(G_losses))
		train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

		sample_z=np.random.uniform(-1,1,size=(1, 1, 1, 100))
		gen_sample=sess.run(generator(z, reuse=True), feed_dict={z:sample_z})

		gen_samples.append(gen_sample)

plt.imshow(gen_samples[0].reshape(64, 64))
plt.show()
plt.imshow(gen_samples[epochs-1].reshape(64, 64))
plt.show()
plt.plot(train_hist['D_losses'])
plt.plot(train_hist['G_losses'])
plt.plot(train_hist['per_epoch_ptimes'])
