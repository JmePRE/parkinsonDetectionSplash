from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from skimage import feature
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
import tensorflow as tf
import cv2
import os
import logging
import numpy as np
from model.color2bw import c2bw
img_width, img_height = 300, 300
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Neural:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        # the folder in which the model and weights are stored
        self.model_folder = os.path.join(os.path.abspath("src"), "static")
        self.create_model()
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                logging.info("neural network initialised")

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(30276,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(48))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(24))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        try:
            self.model.load_weights('model/hgo_model.h5')
        except OSError:
            self.model.load_weights('hgo_model.h5')
        return True


    def quantify_image(self, image):
        # compute the histogram of oriented gradients feature vector for
        # the input image
        s=[]
        features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1")
        # return the feature vector
        s.append(features)
        return s


    def ip1(self, filenum):
        filename = str(filenum)
        print(filename)
        im = c2bw(filename)
        im.save(filename)
        image = cv2.imread(filename, 0)
        if image is None:
            print("lolfuck")
        image = cv2.resize(image, (300, 300))

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('i', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # quantify the image
        features = self.quantify_image(image)
        s = np.array(features)
        with self.graph.as_default():
            l = self.model.predict(s)
        return(l[0][0])

# print(ip1('for_eval\\00075x.png'))
