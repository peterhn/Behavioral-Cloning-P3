import json
from keras.models import Sequential, model_from_json
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
    '''    
    # normalize -1<>+1
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70,25), (0,0))))

    # Conv Layer #0 (depth=3, kernel=1x1) - change color space
    model.add(Convolution2D(3, 1, 1, border_mode='same'))

    # Conv Layer #1 (depth=24, kernel=5x5)
    model.add(Convolution2D(24, 5, 5, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # Conv Layer #2 (depth=36, kernel=5x5)
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # Conv Layer #3 (depth=48, kernel=3x3)
    model.add(Convolution2D(48, 3, 3, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # Conv Layer #4 (depth=64, kernel=3x3)
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Hidden Layer #1
    model.add(Dense(100))
    model.add(ELU())

    # Hidden Layer #2
    model.add(Dense(50))
    model.add(ELU())

    # Hidden Layer #3
    model.add(Dense(10))
    model.add(ELU())

    # Answer
    model.add(Dense(1)) 
    '''
    model = Sequential()
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
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))
    
    return model

driving_log = pd.read_csv('sample_training_data/driving_log.csv')
nb_classes = len(driving_log)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        steering_correction = 0.12
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
		
                name_right = batch_sample[2]
                right_image = cv2.imread(name_right.strip())
                right_angle = center_angle - steering_correction
 
                if np.random.sample() > 0.5:
                    left_image = cv2.flip(left_image, 1)
                    left_angle = left_angle * -1.0
                #if np.random.sample() > 0.5:
                    #right_image = cv2.flip(right_image, 1)
                    #right_angle = right_angle * -1.0
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)
def run():
    try:
        with open('model.json', 'r') as jfile:
	        model = model_from_json(json.load(jfile))

        # import weights
        model.load_weights('model.h5')

        print("Imported model and weights")
    except:
          print('Loading new model')
          model = NvidiaNet()

    samples = []
    for i in range(nb_classes):
        samples.append(driving_log.ix[i])
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Training Samples Length: ', len(train_samples))
    train_generator = generator(train_samples, batch_size=256)
    validation_generator = generator(validation_samples, batch_size=256)

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 3, nb_epoch=10)

    model.save('model.h5')
    model_data = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(model_data, outfile)

run()
