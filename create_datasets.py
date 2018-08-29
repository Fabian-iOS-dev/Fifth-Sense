import os
import random
from old_scripts.resize_images import list_dirs
from PIL import Image
from math import floor
# from crop_banknote import crop

#!!!!! naming convention: start with 'ds' and then characterise the dataset in the name
#'ds*/' directories are ignored in Github for overview reasons
ds_name = os.path.join(os.getcwd(), 'ds_normal') #name of the dataset generated

categories = ['normal', 'angle'] #from which categories pictures should be sampled
perc_train = 0.8

banknotes = list_dirs(os.path.join(os.getcwd(),'dataset_v3')) #data comes from this directory

#create tree structure for dataset
try:
    os.makedirs(ds_name)
except FileExistsError:
        print('Directory ',ds_name + ' already exits. Please rename the variable "ds_name" or delete the directory in question.')
        raise

for note in ['500b', '1000b', '2000b', '5000b', '10000b', '20000b',
             '500f', '1000f', '2000f', '5000f', '10000f', '20000f']:
        os.makedirs(os.path.join(ds_name,'train',note))
        os.makedirs(os.path.join(ds_name, 'test', note))


#distinguish train and test at random
for banknote in banknotes:
    print("In progress: ", banknote)
    for category in categories:
        path = os.path.join(banknote, category)
        pictures = [os.path.join(path, picture) for picture in os.listdir(path) if picture[-3:] == 'JPG']
        num_train_pics = floor(len(pictures)*perc_train) #how many pictures are in train?
        train_pics = random.sample(pictures, num_train_pics)
        test_pics = set(train_pics).symmetric_difference(pictures)

        # sort pictures into train and test
        for pic in train_pics:
            img = Image.open(pic)
            file_name = os.path.split(pic)
            bank_note_str = os.path.split(banknote)
            save_location = os.path.join(ds_name, 'train', bank_note_str[1], file_name[1])
            img.save(save_location)

            # crop(pic, save_location, bank_note_str[1])


        for pic in test_pics:
            img = Image.open(pic)
            file_name = os.path.split(pic)
            bank_note_str = os.path.split(banknote)
            save_location = os.path.join(ds_name, 'test', bank_note_str[1], file_name[1])
            img.save(save_location)

            # crop(pic, save_location, bank_note_str[1])
