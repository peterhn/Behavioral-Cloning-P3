# P3 Behavioral Cloning

**Behavioral Cloning Project**

## Files in this zip
---
* ```model.py ``` script to train and save model
* ```drive.py``` script to tell car how to steer in simulator
* ```model.json``` model architecture
* ```model.h5``` model weights
* ```writeup_report.md```
* ```run.mp4``` video of
## Dataset
---
### 1. Dataset Characteristics
The characteristics of the data was based on the default implementation of the simulator's capturing of data. Which in this case was a frame-by-frame capture of the car driving along the track with certain metadata collected. It included:
- Image from the center of the camera in front of the car
- Image from the left camera in front of the car
- Image from the right camera in front of the car
- Throttle
- Speed
- Brake
- Steering angle

### 2. How data was recored
Originally, I had recorded my own data, but ended up using Udacity's sample data as it proved a much better training set.

My original methodology of collecting data were as follows:
1. Driving around an entire lap: This first step was to driving around the lap as normally as possible with no error and attempted to stay in the center
2. Driving around an entire lap with errors: This next collection method was to drive around a lap while occassionally driving towards the edges, but would steer the car back to the center, to the model can learn what to do if the car starts to drive off course
3. Driving around an entire lap backwards: Driving around an entire lap backwards helped generalize the data
4. Carefully driving around corner: Carefull driving around corners so the model stays inside the corners nicely

## Solution Design
---
### 1. Training
The model was trained using the center, left, and right images. Using the images as input (X) and the corresponding steering angles as the variable to predict (y). Around 19,000 data points were trained this way and some of the data had to be split to prevent overfitting and to be used as a validation set.

The model was trained and validated on sample dataset so it might simply fit to that data. There were other techniques to reduce overfitting in the model, including randomly flipping left biased image to remove the left turn bias of the track.

In order to reduce memory usage, generator was used to feed batches of 256 training examples.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

### 2. Appropriate training data

Originally training data was captured by hand using WASD on the keyboard. There was several methods used when capturing my own data, such as carefully going around corners and ensuring that the turn was smooth, driving backwards around the track to generalize the data, driving towards the edges of the track steering back to the center in order to teach the car what to do if it that situation arises, and driving around in the center of the track.

## Model Architecture and Training Strategy

### 1. Solution Design Approach
The goal I had in mind was to have the car drive within the lane lines, use data which attempted to generalize the car to steer towards the center.

In order to achieve this goal, the overall strategy of the solution was collect correct data necessary to train the car to drive around the track, along with some data that would help the car steer back into the center of the track if it drives off course. 

##### Pre-processing
Originally, the data was recorded with size 160x320 pixels with 3 channels (RGB).

But, when the images were fed into the model they were:
- Cropped: The cropping would help reduce the amount of memory needed to train the data along with remove unecessary parts of the images that weren't needed for training.
-  Normalize: normalized data emperically produce better models and generalize the data in certain changes, ie: color brightness.

##### Designing Model Architecture
###### Convolutions
- Convolutions seemed like a great technique for image processing. It helped generalize certain features of lines in the images which were useful for detecting lane lines
###### Activations
- Choosing ReLU as my activation function was useful for speeding up training and prevented a vanishing gradient.

###### Fully connected layer
- Since this was a regression problem, a 1-neuron final layer was used to output trained data.

## Model Architecture
My model was based off of Nvidia research on deep learning network for applications in self driving cars. It consists of a convolution neural network with 5x5 and 3x3 filter sizes with between 24 and 64. (model.py lines 24 - 28)

The model also includes RELU layers to introduce nonlinearity, and data is normalized using a Keras lambda layer.

The model also crops out the tops and bottoms of each image in order to save memory and only train on the parts of image that mattered.

Here's a snapshop of the final model:
~~~~
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
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
~~~~


