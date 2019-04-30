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

sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

batch_size = 10000

dl = DataLoader("random")
x_train, y_train = dl.get_batch(batch_size)
x_test, y_test = dl.get_batch(batch_size)

model = ACGAN_Model(x_train, y_train, x_test, y_test)
model.train("testingFood")
