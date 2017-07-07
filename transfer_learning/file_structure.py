# The goal of this script is to reorganize the data in the train folder:
"""
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
"""
import os
import numpy as np
from shutil import copyfile

train_files = os.listdir("data/train/")

cat_files = []
dog_files = []

for f in train_files:
	if f.startswith("cat"):
		cat_files.append(f)
	else:
		dog_files.append(f)

# split images into test and validation set

# validation set size
validation_size = 0.2

num_cat_val = len(train_files)*0.2/2
num_dog_val = len(train_files)*0.2/2

cat_val_imgs = np.random.choice(cat_files, int(num_cat_val))
dog_val_imgs = np.random.choice(dog_files, int(num_dog_val))

os.makedirs("data/new_format/train/cats/")
os.makedirs("data/new_format/train/dogs/")
os.makedirs("data/new_format/validation/cats/")
os.makedirs("data/new_format/validation/dogs/")

# iterate over all the training files and move them to the right directory

for f in train_files:
	if f in cat_files:
		if f in cat_val_imgs:
			copyfile("data/train/"+f, "data/new_format/validation/cats/"+f)
		else:
			copyfile("data/train/"+f, "data/new_format/train/cats/"+f)
	else:
		if f in dog_val_imgs:
			copyfile("data/train/"+f, "data/new_format/validation/dogs/"+f)
		else:
			copyfile("data/train/"+f, "data/new_format/train/dogs/"+f)
