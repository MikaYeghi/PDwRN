"""
INPUT DATA FORMAT
Before running this script, make sure that the raw data is stored in the "raw_data" folder in the root directory in the 
following format:

raw_data
├── annotations
└── images

Note that the "annotations" and "images" folders need to contain parallel files, i.e. annotations x.pkl corresponds to the image x.jpg and vice versa.

OUTPUT DATA FORMAT
Make sure that the following tree exists in the root folder:

data
├── test
│   ├── annotations
│   └── images
└── train
    ├── annotations
    └── images

You can run the following sequence of commands in the root folder to create the tree above:
1) mkdir data
2) cd data
3) mkdir train test
4) cd train
5) mkdir annotations images
6) cd ../test
7) mkdir annotations images
"""

import os
import shutil
from tqdm import tqdm

# Variables
raw_data_path = "raw_data"
data_path = "data"
    
# Get the list of filenames. NOTE: this operation assumes that the annotation filenames (excluding their extensions) have
# analogs in among images
filenames = os.listdir(os.path.join(raw_data_path, "annotations"))
for i in range(len(filenames)):
    filenames[i] = filenames[i].split('.')[0]

# Get the set of zones
patches = set()
for filename in filenames:
    patches.add(filename.split('_')[0])
patches = sorted(patches)
    
# The very first patch is the test patch. All other patches are the train patches.
test_patches = patches[:1]
train_patches = patches[1:]

# Move all the train and test files from raw_data folder to the relevan subdirectories in the data folder
for filename in tqdm(filenames):
    src_anns_file = os.path.join(raw_data_path, "annotations", filename + '.pkl')
    src_img_file = os.path.join(raw_data_path, "images", filename + '.jpg')
    
    # Check if it's a train or test sample
    if filename.split('_')[0] in test_patches:
        trg_anns_file = os.path.join(data_path, "test", "annotations")
        trg_img_file = os.path.join(data_path, "test", "images")
    else:
        trg_anns_file = os.path.join(data_path, "train", "annotations")
        trg_img_file = os.path.join(data_path, "train", "images")

    # Copy the data
    shutil.copy(src_anns_file, trg_anns_file)
    shutil.copy(src_img_file, trg_img_file)
    