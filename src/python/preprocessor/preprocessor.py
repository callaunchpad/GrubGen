#!/usr/bin/env python
# coding: utf-8

# In[21]:


from PIL import Image
import glob
import os
import numpy as np
import os


# In[22]:


'''
params 
    main directory "./UPMC_Food101/images/train" that the sub-folders are in 
return
    dictionary containing the sub-folders and a list of all the image paths for a specific food
        keys are "baklava" 
        values are "./UPMC_Food101/images/train/baklava/baklava_808.jpg"
'''
def create_food_dictionary(main_directory, food_types): 
    for root, dirs, files in os.walk(main_directory, topdown=False):
        food_name = root.split('/')[-1] # get the last string of something like "./UPMC_Food101/images/train/baklava"
        if food_name not in food_types:
            food_types[food_name] = []
        for root2, dirs, files in os.walk(root, topdown=False):
            for name in files:
                food_types[food_name].append(os.path.join(root2, name))
    return food_types


# In[23]:


'''
params 
    a list of image paths like ["./UPMC_Food101/images/train/baklava/baklava_808.jpg"]
returns 
    all the images in that list stacked. Each row is a flattened image that contains (R,G,B)
'''
def create_stacked_images(img_list): 
    final_array = [] 
    for image_path in img_list:
        im=Image.open(image_path)
        cropped_image = crop(im, 64, 64)
        final_array.append(rgb(cropped_image, 64, 64)) 
    return final_array      


# In[24]:


# returns an array of rgb tuples for one image 
# returns the flattened image
def rgb(image, width, length):
    flattened = []
    for i in range(width): 
        for j in range(length):
            new = image.load()[i, j]
            flattened.append(new)
    return np.array(flattened)


# In[25]:


# takes in an PIL Image object 
# returns an Image object that is cropped 
def crop(im, new_width, new_height):
    width, height = im.size 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return im.crop((left, top, right, bottom))


# In[26]:


# take in an array and output it to a numpy file 
def create_np_file(array, name):
    np.save(name, array)
    return


# In[27]:


# creates .npy files for each array in dictionary, saves them in the folder specified by the path to folder param 
# saving in folder "food_lists" = './food_lists'
# takes care of cropping of all the images specified in the dictionary 
def create_files(food_dict, path_to_folder): 
    for key, value in food_dict.items():
        array = create_stacked_images(food_dict[key])
        if not os.path.isfile(path_to_folder + '/' + key + '.npy'): 
            create_np_file(array, path_to_folder + '/' + key + '.npy')
    return 


# In[32]:


final_food_dict = {} 
final_food_dict = create_food_dictionary("../../../resources/images/train", final_food_dict) 
final_food_dict = create_food_dictionary("../../../resources/images/test", final_food_dict)


# In[36]:


create_files(final_food_dict, '../../../resources/processed')


# In[63]:


# test = create_stacked_images(final_food_dict["baklava"])


# In[68]:


# create_np_file(test, "./np_array_outputs/baklava.npy")


# # Old Code

# In[101]:


# # crop at the center 64, both train and test 
# # need to find a way to import all images in a directory 

# image_list = []
# cropped_rgb_values = []
# # images stores all the paths of images in the apple_pie folder 
# for path in images: 
#     for filename in glob.glob(path): # for all the images in the apple_pie folder 
#         im=Image.open(filename)
#         image_list.append(im)
#         pix = im.load()

#         #crop it after
#         # Get the RGBA Value of the a pixel of an image
# #         pix[x,y] = value  # Set the RGBA Value of the image (tuple)
#     #     im.save('alive_parrot.png') 
#         image = crop(im, 64, 64)
#         array = rgb(image, 64, 64)


# In[ ]:




