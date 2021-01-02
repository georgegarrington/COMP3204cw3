from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
import matplotlib.pyplot as plt
import pandas
import datetime as dt
import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.engine.topology import Layer
import keras.callbacks as K
from collections import deque
import winsound
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import keras
import os
import pandas
import time
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization
from keras.models import Sequential
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from keras.layers.merge import Concatenate
from keras.layers import Input
from numpy import newaxis
from keras.models import Model
from keras.utils import plot_model
from keras_self_attention import SeqSelfAttention
from sklearn import preprocessing
from sklearn import metrics
import random
from keras.engine.topology import Layer
from keras.initializers import *
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.engine.input_layer import Input
from keras import backend as K
from keras.models import Model
from imutils import paths



# define cnn model
def define_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(15, activation='softmax'))
    conv_base.trainable = False 
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['acc'])
    return model


model = define_model()

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators
train_it = datagen.flow_from_directory('training/',
    class_mode='categorical', batch_size=128, target_size=(224, 224))
test_it = datagen.flow_from_directory('testing/',
    class_mode='categorical', batch_size=128, target_size=(224, 224))


epochs = 2


# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
    validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.000075),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00005),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.000025),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0000075),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))



model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.000005),  metrics=['acc'])
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))



from imutils import paths
imagePaths = list(paths.list_images('\\testing'))

for test_image in imagePaths:
   print("d")
   result = model.predict(test_image)
   print(result)


test_generator = datagen.flow_from_directory(
    directory='tst',
    target_size=(224,224),
    color_mode="rgb",
    batch_size=32,
    class_mode=None,
    shuffle=False
)


test_generator.reset()


pred=model.predict_generator(test_generator,verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)


labels = (test_it.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


filenames=test_generator.filenames

l = []
for f in filenames:
    a = f[8:]
    l.append(a)

results2=pd.DataFrame({"Filename":l,
                      "Predictions":predictions})



print(results2)





















