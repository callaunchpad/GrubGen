import sys

sys.path.insert(0, "../../dataloader")
from dataloader import get_batch, load_files

load_files()

gen_batch, gen_batch_onehot = get_batch(batch_size)
gen_batch = batch_images = np.reshape(gen_batch, [-1, 64, 64, 3])
for onehot in gen_batch_onehot:
    disc_batch, disc_one_hot = get_batch_type(batch_size, onehot)
