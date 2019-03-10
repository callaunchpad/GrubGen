#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from os.path import expanduser
import numpy as np
import random


# In[9]:


path = "..\\..\\..\\resources\\processed" #change to npy files UPMC_Food101/images/train
# dir = os.path.dirname(__file__)
# filename = os.path.join(dir, '/relative/path/to/file/you/want')
# path = "..\\..\\..\\resources\\preprocessed" #change to npy files

food_map = {} # maps index for onehot vector to food file path
food_names = {} # maps index to food name

index = 0
print(os.getcwd())
for file_name in os.listdir(path):
    print(file_name)
    food_map[index] = path + "/" + file_name #string: path of file name
    food_names[index] = file_name
    index += 1

print(food_map)

one_hot = np.array([0 for file in os.listdir(path)])

def get_batch(size):
    batch_one_hots = []
    batch = np.zeros((size, 64*64*3))
    for i in range(0, size):
        batch_vector = one_hot.copy()
        img, cat_index = random_gen()
        batch[i] = img
        batch_vector[cat_index] = 1
        batch_one_hots.append(batch_vector)

    return batch, batch_one_hots


def random_gen():
    # random category
    cat_index = random.randint(0, len(food_map) - 1)
    cat_file_name = np.load(food_map[cat_index])
    cat_file_name = np.reshape(cat_file_name, (cat_file_name.shape[0], -1))
    # print(cat_file_name.shape)

    # random image from category
    img_index = random.randint(0, cat_file_name.shape[0])

    #img is in pixels, of size 3*64*64 by 1
    img = cat_file_name[img_index, :]
    return img, cat_index


b, boh = get_batch(30)
print(food_map)
print(food_names)
print(b)
print(boh)

# {0 : ("apple_pie", __)}

#matplotlib.show(numpy )
