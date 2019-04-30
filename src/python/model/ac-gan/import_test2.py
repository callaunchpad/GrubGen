#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from matplotlib import pyplot as plt



sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

if __name__ == '__main__':
    d = DataLoader("random")
    b, boh = d.get_batch(1)
    count = 0
    for img, oh in zip(b, boh):
        print(img.shape, oh)
        plt.imshow(img)
        count += 1
        plt.show()

    d = DataLoader("cat")
    b, boh = d.get_batch_type(2, 1)
    count = 0
    for img, oh in zip(b, boh):
        print(img.shape, oh)
        plt.imshow(img)
        count += 1
        plt.show()

    b, boh = d.get_batch_type(2, 30)
    count = 0
    for img, oh in zip(b, boh):
        print(img.shape, oh)
        plt.imshow(img)
        count += 1
        plt.show()
