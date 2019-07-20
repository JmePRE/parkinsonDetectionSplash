from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from skimage import feature
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
import cv2
import os
import numpy as np
img_width, img_height = 300, 300
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for
    # the input image
    s=[]
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    # return the feature vector
    s.append(features)
    return s


def ip1(filenum):
    image = cv2.imread('for_eval/'+str(filenum)+'.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (300, 300))

    # threshold the image such that the drawing appears as white
    # on a black background
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # quantify the image
    features = quantify_image(image)
    tmodel = create_model()
    tmodel.load_weights('hgo_model.h5')
    s = np.array(features)
    l = tmodel.predict(s)
    print(l)
    return(l)


print(str("s"))
ip1(1)
ip1(2)
ip1(3)

