from keras.models import Sequential
from keras.layers import Input, Cropping2D, ELU
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
import cv2
import numpy as np

def NvidiaNet():
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160,320,3)))
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


driving_log = pd.read_csv('sample_training_data/driving_log.csv')
nb_classes = len(driving_log)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        steering_correction = 0.11
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = batch_sample[0]
                center_image = cv2.imread(name_center.strip())
                center_angle = float(batch_sample[3])

                # randomly flip the image and angle
                name_left = batch_sample[1]
                left_image = cv2.imread(name_left.strip())
                left_angle = center_angle + steering_correction
                if np.random.sample() > 0.5:
                    left_image = cv2.flip(left_image, 1)
                    left_angle = left_angle * -1.0
                
                name_right = batch_sample[2]
                right_image = cv2.imread(name_right.strip())
                right_angle = center_angle - steering_correction
                if np.random.sample() > 0.5:
                    right_image = cv2.flip(right_image, 1)
                    right_angle = right_angle * -1.0
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)
def run():
    samples = []
    for i in range(nb_classes):
        samples.append(driving_log.ix[i])
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Training Samples Length: ', len(train_samples))
    train_generator = generator(train_samples, batch_size=128)
    validation_generator = generator(validation_samples, batch_size=128)

    model = NvidiaNet()
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 3, nb_epoch=25)
    model.save('model.h5')

run()

''' NON-GENERATOR OPTION
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
        steering_correction = 0.15
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        #list.extend([str(steering_center), str(steering_left), str(steering_right)])
        list.extend([steering_center, steering_left, steering_right])
        #list.extend([steering_center])
        yield list

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

model.fit(X_train, y_train, validation_split=0.2, nb_epoch=25, )
model.save('model.h5')
'''
