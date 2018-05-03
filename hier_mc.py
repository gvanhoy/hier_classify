#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:53:15 2018
Description: 
"""
import numpy as np
import keras
import pickle
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

tiers = [
        ({'name': 'analog',
          'mods': ('fmcw-triangle', 'fmcw-sawtooth', 'am-dsb', 'am-ssb', 'fm')
          },
        {'name': 'digital'})
        ]

tier_1_labels = ['analog', 'digital']

tier_21_labels = ['radar', 'data-bearing']
tier_21_mods = [
        'fmcw-triangle',
        'fmcw-sawtooth',
        'am-dsb',
        'am-ssb',
        'fm'
        ]

tier_22_labels = ['multi-carrier', 'single-carrier']
tier_22_mods = [
        'ofdm-16-bpsk',
        'ofdm-32-bpsk',
        'ofdm-64-bpsk',
        'ofdm-16-qpsk',
        'ofdm-32-qpsk',
        'ofdm-64-qpsk',
        '2gmsk',
        '4gmsk',
        '8gmsk',
        '2fsk',
        '4fsk',
        '8fsk'
        "ook",
        "bpsk",
        "qpsk",
        "4ask",
        "4pam",
        "8psk",
        "8pam",
        "8qam_cross",
        "8qam_circular",
        "16qam",
        "16psk",
        "32qam_cross",
        "32qam_rect",
        "64qam"
        ]

tier_31_labels = ['QAM', 'FSK']
tier_31_mods = [
            "ook",
            "bpsk",
            "qpsk",
            "4ask",
            "4pam",
            "8psk",
            "8pam",
            "8qam_cross",
            "8qam_circular",
            "16qam",
            "16psk",
            "32qam_cross",
            "32qam_rect",
            "64qam"
            '2gmsk',
            '4gmsk',
            '8gmsk',
            '2fsk',
            '4fsk',
            '8fsk'
        ]

tier_41_mods = [
            "ook",
            "bpsk",
            "qpsk",
            "4ask",
            "4pam",
            "8psk",
            "8pam",
            "8qam_cross",
            "8qam_circular",
            "16qam",
            "16psk",
            "32qam_cross",
            "32qam_rect",
            "64qam"
]

tier_42_mods = [
            '2gmsk',
            '4gmsk',
            '8gmsk',
            '2fsk',
            '4fsk',
            '8fsk'
]

tier_43_mods = [
            'ofdm-16-bpsk',
            'ofdm-32-bpsk',
            'ofdm-64-bpsk',
            'ofdm-16-qpsk',
            'ofdm-32-qpsk',
            'ofdm-64-qpsk',
]

tier_44_mods= [
        'fm',
        'am-dsb',
        'am-ssb'
]

tier_45_mods = [
        'fmcw-triangle',
        'fmcw-sawtooth'
]

# %% Get the data and prepare it as necessary
x = []
lbl = []

# Code for generating data. Requires GNU Radio and other code
# data = {'bpsk': [exemplar1, exemplar2, ...,]
#         'qpsk': [exemplar1, exemplar2, ...,]}

mods, data = load_data(channel_type='over_the_air_selective', # "awgn" for adding noise
                       snr_db=10,         # if noise is present
                       num_cplx_samples=NUM_FEATURES, # real + imag
                       num_exemplars=5000) # exemplars per mod
# %%
# f = open(FILE_PATH, "rb")
# mods, data = pickle.loads(f.read())

# Go through the list of modulations and create numeric labels
# this is currently done in a way to support a unique label
# per SNR or multi-labelling later 
# %% Top-level

for tier in tiers:
    for mod in mods:
        if mod in tier['mods']:
            x.append(data[mod])
            for i in range(data[mod].shape[0]):
                if tier['']
                    lbl.append('analog')
                else:
                    lbl.append('digital')

super_class = ['analog', 'digital']
'''
# %% Single vs Multi-carrier
'''
for mod in mods:
    if 'fm' not in mod and 'dsb' not in mod and 'ssb' not in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]):
            if 'ofdm' not in mod:
                lbl.append('single-carrier')
            else:
                lbl.append('multi-carrier')

super_class = ['single-carrier', 'multi-carrier']
'''
# %% Data vs Radar
'''
for mod in mods:
    if 'fm' in mod or 'dsb' in mod or 'ssb' in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]):
            if 'fmcw' in mod:
                lbl.append('radar')
            else:
                lbl.append('data-bearing')

super_class = ['radar', 'data-bearing']
'''
# %% QAM vs FSK
'''
for mod in mods:
    if 'ofdm' not in mod and 'fmcw' not in mod and 'am-' not in mod and 'fm' not in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]):
            if 'fsk' in mod or 'gmsk' in mod:
                lbl.append('fsk')
            else:
                lbl.append('qam')

super_class = ['FSK', 'QAM']
'''
# %% AM//FM
'''
for mod in mods:
    if 'am-' in mod or mod == 'fm':
        x.append(data[mod])
        for i in range(data[mod].shape[0]): lbl.append(mod)

super_class = ['am-ssb', 'am-dsb', 'fm']
'''
# %% FMCW
'''
for mod in mods:
    if 'fmcw' in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]): lbl.append(mod)

super_class = ['fmcw-triangle', 'fmcw-sawtooth']
'''
# %% FSK
'''
for mod in mods:
    if 'fsk' in mod or 'gmsk' in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]): lbl.append(mod)

super_class = ['2fsk', '4fsk', '8fsk', '2gmsk', '4gmsk', '8gmsk']
'''
# %% QAM
'''
for mod in mods:
    if 'fsk' not in mod and 'gmsk' not in mod and 'am-' not in mod and 'fmcw' not in mod and 'ofdm' not in mod and 'fm' not in mod:
        x.append(data[mod])
        for i in range(data[mod].shape[0]): lbl.append(mod)

super_class = ["ook", "bpsk", "qpsk", "4ask", "4pam", "8psk", "8pam", 
               "8qam_cross", "8qam_circular", "16qam", 
               "16psk", "32qam_cross", "32qam_rect", "64qam"]

x = np.vstack(x) # stack it up my friend
x_train, x_test, y_train, y_test = train_test_split(x, lbl, test_size=0.33, random_state=42)

y_train = keras.utils.to_categorical(map(lambda x: super_class.index(y_train[x]), range(len(y_train))))
y_test = keras.utils.to_categorical(map(lambda x: super_class.index(y_test[x]), range(len(y_test))))

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
plot_confusion_matrix(cnf_matrix, classes=super_class, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

