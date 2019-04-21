#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, mode="random", shuffle=True):
        print("Initializing", mode, "dataloader")
        assert mode=="random" or mode == "cat", ("Unrecognized mode,", mode)
        self.mode = mode
        self.resources = self.find_resources_path()
        self.path = self.resources + "/processed"

        self.load_files()

        if shuffle:
            self.shuffle_data()

        #Compute batch pointers
        if self.mode == "random":
            self.curr = 0
        elif self.mode == "cat":
            self.curr_lst = [0] * len(self.images_lst)

        print("Done initializing", mode, "dataloader")

    def shuffle_data(self):
        if self.mode == "random":
                self.num_pts = self.images.shape[0]
                p = np.random.permutation(self.num_pts)
                print("Total of", self.num_pts, "images loaded into memory")
                self.images = self.images[p]
                self.onehots = self.onehots[p]

        elif self.mode == "cat":
            for i in range(len(self.images_lst)):
                num = self.images_lst[i].shape[0]
                p = np.random.permutation(num)
                self.images_lst[i] = self.images_lst[i][p]

    def load_files(self):
        #loads the .npy files
        index = 0
        temp_imgs = []
        
        for file_name in os.listdir(self.path):
            if file_name.endswith(".npy"):
                file_path = self.path + "/" + file_name
                data = np.load(file_path)
                temp_imgs.append(data)
                print("Loaded", file_name, "\t\t with index", index, "and", data.shape[0], "imgs")
                index += 1

        temp_onehot = []
        for i, data in enumerate(temp_imgs):
            num_pts = data.shape[0]
            one_hot = np.zeros(index)
            one_hot[i] = 1
            one_hots = np.tile(one_hot, [num_pts, 1])
            temp_onehot.append(one_hots)

        if self.mode == "random":
            self.images = np.vstack(temp_imgs)
            self.onehots = np.vstack(temp_onehot)
        elif self.mode == "cat":
            self.images_lst = temp_imgs
            self.one_hots_lst = temp_onehot #hope is that more memory storage but faster speed

    def get_batch(self, size):
        if self.mode == "cat":
            print("This dataloader was configured as a categorical loader, does not support get_batch")
            raise Exception

        if self.curr + size >= self.num_pts:
            self.curr = 0

        ret = self.images[self.curr:self.curr + size], self.onehots[self.curr:self.curr + size]
        self.curr += size
        return ret

    def get_batch_type(self, size, cat_index):
        if self.mode == "random":
            print("This dataloader was configured as a random loader, does not support get_batch_type")
            raise Exception

        cat_num_pts = self.images_lst[cat_index].shape[0]
        curr = self.curr_lst[cat_index]
        if curr + size >= cat_num_pts:
            curr = 0
            self.curr_lst[cat_index] = 0

        ret = self.images_lst[cat_index][curr:curr + size], self.one_hots_lst[cat_index][0:size]
        self.curr_lst[cat_index] += size
        return ret

    def find_resources_path(self):
        cwd = os.getcwd()
        gg_idx = cwd.index("src")
        new_wd = cwd[gg_idx:]
        num_slash = max(new_wd.count("\\"), new_wd.count("/")) + 1 #max to account for windows vs unix
        pathing = "../" * num_slash + "resources"
        return pathing