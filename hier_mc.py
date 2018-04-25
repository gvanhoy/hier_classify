#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:53:15 2018
Description: 
"""
import numpy as np
import keras
import urllib2
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from classify.mc_dataset import load_data # necessary to gen aribtrary data

BATCH_SIZE = 128
EPOCHS = 10
NUM_FEATURES = 128
DATA_URL = "https://s3.amazonaws.com/radio-machine-learning/mod_14_clean.pkl"
FILE_PATH = "/home/gvanhoy/mod_14_clean.pkl"

# %% Get the data and prepare it as necessary
x = []
lbl = []

# Code for generating data. Requires GNU Radio and other code
'''
# data = {'bpsk': [exemplar1, exemplar2, ...,]
#         'qpsk': [exemplar1, exemplar2, ...,]}

mods, data = load_data(channel_type=None, # "awgn" for adding noise
                       snr_db=20,         # if noise is present
                       num_cplx_samples=NUM_FEATURES/2, # real + imag
                       num_exemplars=5000) # exemplars per mod
'''

f = open(FILE_PATH, "rb")
mods, data = pickle.loads(f.read())

# Go through the list of modulations and create numeric labels
# this is currently done in a way to support a unique label
# per SNR or multi-labelling later
for mod in mods:
    x.append(data[mod])
    for i in range(data[mod].shape[0]):  
        lbl.append(mod)
    
x = np.vstack(x) # stack it up my friend
x_train, x_test, y_train, y_test = train_test_split(x, lbl, test_size=0.33, random_state=42)

y_train = keras.utils.to_categorical(map(lambda x: mods.index(y_train[x]), range(len(y_train))))
y_test = keras.utils.to_categorical(map(lambda x: mods.index(y_test[x]), range(len(y_test))))

# %% Make the model
'''
So this is currently a 2 stage convolutional network. The first 
stage is only a filter size 2 in the hopes of capturing the
unique transitions that each modulation makes from level
to level.
The second stage is there to capture the set of possible 
transitions that are unique to each modulation.
Everything else is standard/a guess.
'''
in_shp = list(x_train.shape[1:])
model = Sequential()
model.add(Conv1D(filters=256,
                 kernel_size=2,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 name="conv1", 
                 kernel_initializer='glorot_uniform',
                 input_shape=in_shp))
model.add(Dropout(.5))
model.add(MaxPooling1D(pool_size=1,padding='valid', name="pool1"))
model.add(Conv1D(filters=264,
                 kernel_size=8,
                 strides=4,
                 padding='valid', 
                 activation="relu", 
                 name="conv2", 
                 kernel_initializer='glorot_uniform',
                 input_shape=in_shp))
model.add(Dropout(.5))
model.add(MaxPooling1D(pool_size=1, padding='valid', name="pool2"))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(.5))
model.add(Dense(len(mods), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))

# TODO: It would be nice to pickle up the model here just in case

# %% Visualize the results. Can haz confusion matrix?
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
