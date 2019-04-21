import numpy as np
from PIL import Image

pics = np.load("baklava.npy")

np.save('baklava_test2', pics[2])