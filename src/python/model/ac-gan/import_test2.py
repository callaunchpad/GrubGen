#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from matplotlib import pyplot as plt


sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

if __name__ == '__main__':
    d = DataLoader()
    # testing
    b, boh = d.get_batch(30)
    # to show images in batch:
    count = 0
    for img in b:
        # print(img.shape, img.dtype)
        # img = np.reshape(i, (64, 64, 3))
        # img /= 255
        plt.imshow(img)
        count += 1

        plt.show()