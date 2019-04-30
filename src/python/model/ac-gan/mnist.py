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
from ACGAN import ACGAN_Model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims([scipy.misc.imresize(i, (64, 64, 1)) for i in x_train], axis=3)
x_test = np.expand_dims([scipy.misc.imresize(i, (64, 64, 1)) for i in x_test], axis=3)
batch_size=200
digits = [i for i in range(10)]


y_train = ACGAN_Model.one_hot(y_train)
y_test = ACGAN_Model.one_hot(y_test)

model = ACGAN_Model(x_train, y_train, x_test, y_test)
model.train("testingClass", 1)
epochs=100
