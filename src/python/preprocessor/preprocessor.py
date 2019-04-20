#!/usr/bin/env python
from PIL import Image
import glob
import os
import numpy as np
import os
from PIL import Image

resources = "../../../resources"


def is_img_name(name):
    if name.endswith("png"):
        print("unexpected png, please add feature")
        1/0
    return name.endswith("jpg")

'''
params 
    main directory "./UPMC_Food101/images/train" that the sub-folders are in 
return
    dictionary containing the sub-folders and a list of all the image paths for a specific food
        keys are "baklava" 
        values are "./UPMC_Food101/images/train/baklava/baklava_808.jpg"
'''
def create_food_dictionary(main_directory, food_types):
    for subdir in os.listdir(main_directory):
        food_name = subdir
        if food_name not in food_types:
            food_types[food_name] = []
        for food_img_path in os.listdir(os.path.join(main_directory, food_name)):
            if is_img_name(food_img_path):
                food_types[food_name].append(os.path.join(main_directory, food_name, food_img_path))
    return food_types


'''
params 
    a list of image paths like ["./UPMC_Food101/images/train/baklava/baklava_808.jpg"]
returns 
    all the images in that list stacked. Each row is a flattened image that contains (R,G,B)
'''
def create_stacked_images(img_list, size): 
    final_array = [] 
    for image_path in img_list:

        im = Image.open(image_path)
        # imgSize = im.size
        # rawData = im.tobytes()
        # im = Image.frombytes('F', im.size, im.tobytes(), 'raw')
        # raw = open(image_path, 'rb').read()
        # im = Image.frombytes('F', imgSize, raw)
        # print(img_arr[0][0][0], resized.getpixel((0,0))[0])
        # print(img_arr.dtype)
        cropped_image = crop(im)
        resized = resize(cropped_image, size)
        img_arr = np.array(resized)
        
        final_array.append(img_arr)
    return np.stack(final_array)      

# returns an array of rgb tuples for one image 
# returns the flattened image
def rgb(image, width, length):
    1/0 #don't use this
    flattened = []
    for i in range(width): 
        for j in range(length):
            new = image.load()[i, j]
            flattened.append(new)
    return np.array(flattened)

# takes in an PIL Image object 
# returns an Image object that is cropped 
# crops to square 
def crop(im):
    width, height = im.size
    if width > height: 
        left = (width - height)//2
        top = 0
        right = (width + height)//2
        bottom = height
    else: 
        left = 0
        top = (height - width)//2
        right = width
        bottom = (height + width)//2
    im = im.crop((left, top, right, bottom))
    assert im.size[0] == im.size[1], im.size
    return im


def resize(im, size):
    im = im.resize((size, size), Image.ANTIALIAS)
    return im


# take in an array and output it to a numpy file 
def create_np_file(array, name, overwrite=False):
    if overwrite or not os.path.isfile(name): 
        print(name, "written")
        np.save(name, array)
    else:
        print("Already exists", name, "skipping...")

# creates .npy files for each array in dictionary, saves them in the folder specified by the path to folder param 
# saving in folder "food_lists" = './food_lists'
# takes care of cropping of all the images specified in the dictionary 
def create_files(food_dict, path_to_folder, size, overwrite): 
    for key, food_dir_list in food_dict.items():
        if len(food_dir_list) < 1:
            continue
        array = create_stacked_images(food_dir_list, size)
        create_np_file(array, path_to_folder + '/' + key + '.npy', overwrite)

if __name__ == '__main__':
    import sys
    assert len(sys.argv) < 4
    size = 64
    overwrite = False
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) == 3:
        assert sys.argv[2] =='0' or sys.argv[2] =='1'
        overwrite = bool(sys.argv[2])
        if overwrite:
            print("Overwrite flag on")

    final_food_dict = {} 
    final_food_dict = create_food_dictionary(resources + "/images/train", final_food_dict) 
    final_food_dict = create_food_dictionary(resources + "/images/test", final_food_dict)

    create_files(final_food_dict, resources + "/processed", size, overwrite)



