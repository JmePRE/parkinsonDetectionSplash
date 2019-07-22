from PIL import Image # used for loading images
import numpy as np
import random
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from skimage import feature
from imutils import paths
import cv2
from keras.applications import VGG16
import tensorflow as tf
img_width, img_height = 300, 300
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
sess = tf.Session(config=config)

train_dir = 'training_data/training'
test_dir = 'training_data/testing'

nb_train_samples = 1024
nb_validation_samples = 128
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(64, (4, 4), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255)
"""rescale=1. / 255,
    rotation_range=10,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1"""
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')

'''
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs, validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('model_saved.h5')
'''

def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    # return the feature vector
    return features

def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        l = 0
        if(label == 'parkinson'):
            l = 1
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (300, 300))

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # quantify the image
        features = quantify_image(image)

        # update the data and labels lists, respectively
        data.append(features)
        labels.append(l)

    # return the data and labels
    return (np.array(data), np.array(labels))


(trainX, trainY) = load_split(train_dir)
(testX, testY) = load_split(test_dir)
s = testX[0].shape
print("s="+str(s))

def create_model():
    tmodel = Sequential()
    tmodel.add(Dense(24, input_shape=(30276,)))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(0.3))
    tmodel.add(Dense(48))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(0.3))
    tmodel.add(Dense(24))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(0.3))
    tmodel.add(Dense(1))
    tmodel.add(Activation('sigmoid'))
    return tmodel


def train():
    tmodel1 = create_model()
    tmodel1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    for i in range(12):
        print(i)
        tmodel1.fit(trainX, trainY, epochs=1, batch_size=8)
        t = tmodel1.evaluate(testX, testY)
        print(t)
        print(tmodel1.metrics_names)
    return tmodel1

print("m.pyrun")
m = train()
m.save_weights('hgo1_model.h5')
