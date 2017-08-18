# Set up the library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import cv2
import os
import math
from PIL import Image         
from os import getcwd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
%matplotlib inline

# Load data from file
lines= []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.05)
        
#np.random.shuffle(lines)
#split_i = int(len(lines) * 0.9)
#X_train, y_train = list(zip(*lines[:split_i]))
#X_valid, y_valid = list(zip(*lines[split_i:]))


#X_train, y_train = np.array(X_train), np.array(y_train)
#X_valid, y_valid = np.array(X_valid), np.array(y_valid) 

# Define the generator function
import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/'+batch_sample[0]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print('Done Generator')


# Training Architecture 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda

from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


model= Sequential()      
model.add(Cropping2D(cropping=((70, 25), (0, 0)),dim_ordering='tf',input_shape=(160, 320, 3)))

# Normalize the training data
model.add(Lambda(lambda x: (x/255.0) - 0.5))
model.add(Convolution2D(16,8,8,activation = "relu"))
model.add(Convolution2D(32,5,5,activation = "relu"))
model.add(Convolution2D(64,5,5,activation = "relu"))
#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('softmax'))


model.compile(loss='mse', optimizer='adam')
#model.fit(myTrain,validation_split =0.2, shuffle=True, nb_epoch=3)
#model.fit(X_train, Y_train, nb_epoch=3,verbose=1, validation_data=(X_valid, Y_valid))

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=15)


model.save('model.h5')
print('Done')