from keras.models import Sequential
from keras.layers import Input, Cropping2D, ELU
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16

from scipy.misc import imread
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import pandas as pd

import cv2, numpy as np

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

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
    model.compile(loss='mse', optimizer='adam')
    return model

def NvidiaNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
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
        #list.extend([im_center])
        yield list

def process_steering_angle():
    list = []
    for i in range(nb_classes):
        steering_center = driving_log.ix[i][3]
        steering_correction = 0.09
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        #list.extend([str(steering_center), str(steering_left), str(steering_right)])
        list.extend([steering_center, steering_left, steering_right])
        #list.extend([steering_center])
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
model = NvidiaNet()
#X_train, X_val, y_train, y_val = train_test_split(car_images, steering_angles, test_size=0.33, random_state=0)
X_train, y_train = shuffle(car_images, steering_angles) 

X_train = np.array(X_train)
y_train = np.array(y_train)

# from sklearn.preprocessing import LabelBinarizer
# label_binarizer = LabelBinarizer()
# y_one_hot = label_binarizer.fit_transform(y_train)

print('Training...')
# model.fit(X_train, y_one_hot, batch_size=128, nb_epoch=2, validation_split=0.2)
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=10)
model.save('model.h5')