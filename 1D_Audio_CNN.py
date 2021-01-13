import os
import h5py
import librosa
import spafe
from spafe.utils.preprocessing import pre_emphasis
from spafe.fbanks import gammatone_fbanks
import itertools
import librosa.display
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
#from summary import summary
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# TRAINING PARAMETERS
hyperP = dict()
hyperP['numEpoch'] = 20
hyperP['learning_rate'] = .0001
hyperP['batchSize'] = 100

# INPUT-HIDDEN_lAYER-OUTPUT DIMENTIONS
L_dim = dict()
L_dim['D_in'] = 44100
L_dim['H_1'] = 16
L_dim['H_2'] = 32
L_dim['H_3'] = 64
L_dim['H_4'] = 128
L_dim['H_5'] = 256
L_dim['H_6'] = 512
L_dim['H_7'] = 1024
L_dim['D_out'] = 10
KERNEL=8

def generator(features, labels, batch_size):
    while True:
        batch_features = []
        batch_labels = []

        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(features),1)
            batch_features.extend(features[index])
            batch_labels.extend(labels[index])
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        yield batch_features, batch_labels

gtzan_dir = 'C:/Users/System_hs/Desktop/data/genres/'
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
savename = "offseti2-duration2.npy"

list_data = []
classes = []
# Get file list from the folders
for x,_ in genres.items():
    folder = gtzan_dir + x
    for root, subdirs, files in os.walk(folder):
        for file in files:
            file_name = folder + "/" + file
            for i in range(0,1,1):
                wavedata, samplerate = librosa.load(file_name, sr=None, mono=True, offset=i, duration=3)
                #wavedata = pre_emphasis(wavedat, pre_emph_coeff=0.95)
                #print("Sample:",samplerate)
                data = np.array(wavedata)
                #print("wavedata:",data)
                #print("Shape",data.shape)
                #input()
                wavedata = wavedata[:, np.newaxis]
                list_data.append(wavedata)
                classes.append(genres[x])
            # Save the file name and the genre

#np.save("list_data-" + savename, list_data)
#np.save("classes-" + savename, classes)

#print(classes)

data = np.array(list_data)

input(data.shape)
print(data)


labelencoder = LabelEncoder()
labelencoder.fit(classes)

n_classes = len(labelencoder.classes_)
print (n_classes, "classes:", ", ",(list(labelencoder.classes_)))

classes_num = labelencoder.transform(classes)


classes_num_1hot = to_categorical(classes_num)

testset_size = 0.20
splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes_num)

for train_index, test_index in splits:
    print("TRAIN INDEX:", train_index)
    print("TEST INDEX:", test_index)

    train_set = data[train_index]
    test_set = data[test_index]

    train_classes = classes_num[train_index]
    train_classes_1hot = classes_num_1hot[train_index]
    test_classes = classes_num[test_index]
    test_classes_1hot = classes_num_1hot[test_index]

print (train_set.shape)
print (test_set.shape)

input_shape = train_set.shape



model = Sequential()
model.add(Conv1D(16, 64, input_shape=input_shape[1:], activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=8, strides = 8))
model.add(Conv1D(32, 32, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=8, strides = 8))
model.add(Conv1D(64, 16, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(Conv1D(128, 8, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(Conv1D(256, 4, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4, strides = 4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy' , optimizer='Adadelta' , metrics=['accuracy'])
model.summary()



batch_size = 20
History = model.fit_generator(generator(train_set, train_classes_1hot, batch_size), steps_per_epoch=len(train_set),epochs=50, validation_data=generator(test_set, test_classes_1hot, batch_size), validation_steps=20)
