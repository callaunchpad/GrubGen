import tensorflow as tf
import numpy as np
import matplotlib
import sys
import scipy
import random
from sklearn.utils import shuffle
#from ../dataloader.dataloader import get_batch
matplotlib.use('Agg')
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from ACGAN import ACGAN_Model

sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

batch_size = 1000

def add_multiple_classes(classes):
    x_train, y_train = [], []
    for i in classes:
        x_train1, y_train1 = dl.get_batch_type(batch_size, i)
        x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
        if (x_train == []):
            x_train = x_train1
            y_train = y_train1
        else:
            x_train = np.vstack((x_train, x_train1))
            y_train = np.vstack((y_train, y_train1))
    return shuffle(x_train, y_train, random_state=0)


dl = DataLoader("cat")

x_train, y_train = add_multiple_classes([1, 2, 3])
model = ACGAN_Model(x_train, y_train, np.array([]), np.array([]), num_classes=101)
model.train("GrubGen3lrg0.001lrd0.001", epochs=100, lrg=0.001, lrd=0.001)
