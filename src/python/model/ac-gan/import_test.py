import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, "../../dataloader")
from dataloader import DataLoader

d = DataLoader()
d.load_files()

b, boh = d.get_batch(2)
# to show images in batch:
count = 0
for i in b:
    img = np.reshape(i, (64, 64, 3))
    img /= 255
    plt.imshow(img)
    count += 1

    plt.show()

batch_size = 30
gen_batch, gen_batch_onehot = d.get_batch(batch_size)
gen_batch = batch_images = np.reshape(gen_batch, [-1, 64, 64, 3])
for onehot in gen_batch_onehot:
    disc_batch, disc_one_hot = d.get_batch_type(batch_size, onehot)
