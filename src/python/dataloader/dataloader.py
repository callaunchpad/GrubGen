#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import random
from matplotlib import pyplot as plt





class DataLoader:
    def __init__(self):
        self.resources = self.find_resources_path()
        self.path = self.resources + "/processed"
        self.food_paths = [] # maps index for onehot vector to food file path
        self.food_names = [] # maps index to food name

        # print(os.getcwd())

    def find_resources_path(self):
        cwd = os.getcwd()
        gg_idx = cwd.index("src")
        new_wd = cwd[gg_idx:]
        num_slash = new_wd.count("\\") + 1
        pathing = "../" * num_slash + "resources"

        print(new_wd, num_slash)

        return pathing

        
        # "../../../resources"

    def load_files(self):
        #loads the directory of .npy files into array
        index = 0
        for file_name in os.listdir(self.path):
            if file_name.endswith(".npy"):
                # print(file_name)
                self.food_paths.append(self.path + "/" + file_name) # string: path of file name
                self.food_names.append(file_name[0:(len(file_name)-4)])

        # one_hot = np.array([0 for file in os.listdir(path)])
        self.one_hot = np.zeros(len(self.food_paths)).astype(np.int)

        print("FOOD PATHS:")
        print(self.food_paths)
        print("FOOD NAMES:")
        print(self.food_names)

    # print("FOOD PATHS:")
    # print(food_paths)
    # print("FOOD NAMES:")
    # print(food_names)

    

    def get_batch_type(self, size, cat_index):
        # cat_index = list(one_hot.where(1)
        batch_one_hots = []
        cat_file = np.load(food_paths[cat_index])
        batch = np.zeros((size, 64*64*3))
        for i in range(size):
            img_index = random.randint(0, cat_file.shape[0])
            img = cat_file[img_index, :]
            batch[i] = img
            batch_vector[cat_index] = 1
            batch_one_hots.append(batch_vector)
        return batch, batch_one_hots


    def get_batch(self, size):
        batch_one_hots = []
        batch = np.zeros((size, 64*64*3))
        for i in range(0, size):
            batch_vector = self.one_hot.copy()
            img, cat_index = self._random_gen()
            batch[i] = img
            batch_vector[cat_index] = 1
            batch_one_hots.append(batch_vector)
        return batch, batch_one_hots

    def _random_gen(self):
        # random category
        cat_index = random.randint(0, len(self.food_paths)-1)
        cat_file = np.load(self.food_paths[cat_index])
        cat_file = np.reshape(cat_file, (cat_file.shape[0], -1))

        # random image from category
        img_index = random.randint(0, cat_file.shape[0])

        # img is in pixels, of size 3*64*64 by 1
        img = cat_file[img_index]
        return img, cat_index


if __name__ == '__main__':
    d = DataLoader()
    d.load_files()
    # testing
    print(d.one_hot)
    b, boh = d.get_batch(30)
    # to show images in batch:
    count = 0
    for i in b:
        img = np.reshape(i, (64, 64, 3))
        img /= 255
        plt.imshow(img)
        count += 1

        plt.show()
