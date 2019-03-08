#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from os.path import expanduser
import numpy as np
import random


# In[9]:


path = "./UPMC_Food101/images/train" #change to npy files
food_map = {} #maps index for onehot vector to food file name
index=0
for file_name in os.listdir(path):
    food_map[index] = path + "/" + file_name #string: path of file name
    index += 1
    print(file_name)

print(food_map)

one_hot = np.array([0 for file in os.listdir(path)])

def get_batch(size):
    batch = np.zeros((size, 64*64*3))
    batch_vector = one_hot.copy()
    for i in range(0, size):
        img, cat_index = random_gen()
        batch[i] = img
        batch_vector[cat_index] = 1

    return batch, batch_vector


def random_gen():
    # random category
    cat_index = random.randint(0, food_map.size)
    cat_file_name = np.load(food_map[cat_index])

    # random image from category
    img_index = random.randint(0, cat_file_name.shape[0])

    #img is in pixels, of size 3*64*64 by 1
    img = cat_file_name[img_index, :]
    return img, cat_index





# In[ ]:
