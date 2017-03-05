from keras.models import Sequential
from keras.layers import Input, Cropping2D
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16

from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import pandas as pd

import cv2, numpy as np


def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Activation("relu"))
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(77))
    model.add(Dense(1))
    return model

def model2():
    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(160, 320, 3), border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Conv2D(128, 3, 3, border_mode='same', activation='relu'),
        Conv2D(256, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    return model

driving_log = pd.read_csv('training_data/driving_log.csv')
nb_classes = len(driving_log)

def process_car_image():
    list = []
    for i in range(nb_classes):
        image_center_path = driving_log.ix[i][0]
        im_center = imread(image_center_path).astype(np.float32)
        im_center = im_center - np.mean(im_center)

        image_left_path = driving_log.ix[i][1]
        im_left = imread(image_left_path).astype(np.float32)
        im_left = im_left - np.mean(im_left)

        image_right_path = driving_log.ix[i][2]
        im_right = imread(image_right_path).astype(np.float32)
        im_right = im_right - np.mean(im_right)

        #list.extend([np.array(im_center), np.array(im_left), np.array(im_right)])
        list.extend([im_center, im_left, im_right])
        yield list

def process_steering_angle():
    list = []
    for i in range(nb_classes):
        steering_center = driving_log.ix[i][3]
        steering_correction = 0.2
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        #list.extend([str(steering_center), str(steering_left), str(steering_right)])
        list.extend([steering_center, steering_left, steering_right]) 
        yield list

#if __name__ == '__main__':
# load images

car_images = []
steering_angles = []

car_image_processor = process_car_image()
steering_angle_processor = process_steering_angle()

for i in range(nb_classes):
    car_images = next(car_image_processor)
    steering_angles = next(steering_angle_processor)


# model = AlexNet()
# model = VGG_16()
# model = LeNet()
# model.compile(loss='mse', optimizer='adam')
model = model2()
#X_train, X_val, y_train, y_val = train_test_split(car_images, steering_angles, test_size=0.33, random_state=0)
X_train, y_train = shuffle(car_images, steering_angles) 

X_train = np.array(X_train)
y_train = np.array(y_train)

# from sklearn.preprocessing import LabelBinarizer
# label_binarizer = LabelBinarizer()
# y_one_hot = label_binarizer.fit_transform(y_train)

print('Training...')
# model.fit(X_train, y_one_hot, batch_size=128, nb_epoch=2, validation_split=0.2)
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=5)
model.save('model.h5')