from sklearn.datasets import load_files
import numpy as np
import keras.utils as np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from keras.models import Sequential
from keras.layers import MaxPooling2D, Activation, Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# Define the model
simple_model = Sequential()

# Layer 1
simple_model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
simple_model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
simple_model.add(Conv2D(32, (3, 3), activation="relu"))
simple_model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
simple_model.add(Conv2D(64, (3, 3), activation="relu"))
simple_model.add(MaxPooling2D(pool_size=(2,2)))

# Add some FC layers here and then a binary head
simple_model.add(Flatten()) # You can do some fancier stuff like Global Average Pooling (GAP)
simple_model.add(Dense(64, activation="relu"))
simple_model.add(Dropout(0.5)) # dropout with retention probability of 0.5
simple_model.add(Dense(1, activation="sigmoid"))


# Time for some model trainning

opt = Adam() # Adam is usually the go to since it combines the best of RMSProp and Adadelta

simple_model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

batch_size = 16 # small to induce some degree of randomness
n_epochs = 50

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# technically we are not doing any permutations, just rescaling our validation images
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/new_format/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/new_format/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')


# To keep track of the best model
checkpointer = ModelCheckpoint(filepath='models/simple_cnn_best.h5', 
                               verbose=1, save_best_only=True, monitor="val_loss")

history = simple_model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[checkpointer])