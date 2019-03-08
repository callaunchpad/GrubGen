#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os 
from os.path import expanduser
import numpy as np
import random


# In[9]:


path = "./UPMC_Food101/images/train" #change to npy files
food_map = {}
count=0
#food_map: matches index for onehot vector to food file
for file_name in os.listdir(path):
    food_map[count]=path+"/"+file_name #string path
    count += 1
    print(file_name)

print(food_map)

one_hot = np.array([0 for file in os.listdir(path)])

def get_batch(size):
    batch = np.zeros((size, 64*64*3))
    batch_vector = one_hot.copy()
    for i in range(0, size):
        img, cat_ind = random_gen()
        batch[i] = img
        batch_vector[cat_ind] = 1
    
    return batch, batch_vector
        
    
def random_gen():
    # random cat
    cat_ind = random.randint(0, food_map.size)
    cat_file = np.load(food_map[cat_ind])
    
    # random img from cat
    img_ind = random.randint(0, cat_file.shape[0]) 
    
    #in pixels, 3*3*64
    img = cat_file[img_ind, :]
    return img, cat_ind


    


# In[ ]:




