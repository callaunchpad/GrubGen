import tensorflow as tf
import time
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.insert(0, '../../dataloader')
from dataloader import get_batch, load_files
from PIL import Image



batch_images = get_batch(1)[0]


img = Image.fromarray(batch_images, 'RGB')
img.show()