#!/usr/bin/env python
# coding: utf-8

import os
from os.path import expanduser
import numpy as np
import random
from matplotlib import pyplot as plt

path = "../../../resources/processed" # on mac
# path = "..\\..\\..\\resources\\preprocessed" # on windows

food_paths = {} # maps index for onehot vector to food file path
food_names = {} # maps index to food name

# print("current working directory:")
# print(os.getcwd())

index = 0
for file_name in os.listdir(path):
    if ".npy" in file_name:
        # print(file_name)
        food_paths[index] = path + "/" + file_name # string: path of file name
        food_names[index] = file_name[0:(len(file_name)-4)]
        index += 1

# print("FOOD PATHS:")
# print(food_paths)
# print("FOOD NAMES:")
# print(food_names)

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
    cat_index = random.randint(0, len(food_paths) - 1)
    print(cat_index)
    cat_file = np.load(food_paths[cat_index])
    cat_file = np.reshape(cat_file, (cat_file.shape[0], -1))

    # random image from category
    img_index = random.randint(0, cat_file.shape[0])

    # img is in pixels, of size 3*64*64 by 1
    img = cat_file[img_index, :]
    return img, cat_index

# testing
# b, boh = get_batch(30)
# print(food_paths)
# print(food_names)
# print(b)
# print(boh)
#
# to show images in batch:
# count = 0
# for i in b:
#     img = np.reshape(i, (64, 64, 3))
#     img /= 255
#     plt.imshow(img)
#     count += 1
#     break
#
# plt.show()
