#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:53:15 2018
Description: 
"""
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from classify.mc_dataset import load_data # necessary to gen aribtrary data

BATCH_SIZE = 128
EPOCHS = 10
NUM_FEATURES = 128
DATA_URL = "https://s3.amazonaws.com/radio-machine-learning/mod_14_clean.pkl"
FILE_PATH = "/home/gvanhoy/mod_19_snr_10.pkl"

# %% Get the data and prepare it as necessary
x = []
lbl = []

df = pd.read_pickle("/home/gvanhoy/mod_26_rsf.gz")
# f = open(FILE_PATH, "rb")
# mods, data = pickle.loads(f.read())

# Go through the list of modulations and create numeric labels
# this is currently done in a way to support a unique label
# per SNR or multi-labelling later 
# %% Top-level
mods = df["mod_name"].unique().tolist()

for mod in mods:
    for series in df[df["mod_name"] == mod].iloc[:,:NUM_FEATURES].values:
        x.append(series)
        lbl.append(mod)
        
x = np.vstack(x)
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
                 kernel_size=16,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 name="conv1", 
                 kernel_initializer='glorot_uniform',
                 input_shape=in_shp))
model.add(Dropout(.5))
# model.add(MaxPooling1D(pool_size=1,padding='valid', name="pool1"))
model.add(Conv1D(filters=256,
                 kernel_size=8,
                 strides=4,
                 padding='valid', 
                 activation="relu", 
                 name="conv2", 
                 kernel_initializer='glorot_uniform'))
model.add(Dropout(.5))
# model.add(MaxPooling1D(pool_size=1, padding='valid', name="pool2"))
model.add(Conv1D(filters=256,
                 kernel_size=2,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 name="conv3", 
                 kernel_initializer='glorot_uniform'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(.5))
model.add(Dense(len(super_class), kernel_initializer='he_normal', name="dense2" ))
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

# %% Confusion matrix fcn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60)
    plt.yticks(tick_marks, classes)

    fmt = '0.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %% Actual confusion
y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=1)
y_test_2 = np.argmax(y_test, axis=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_2, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=mods, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

