import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

def crop_image(img):
    pass

def generate_image():
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

        list.append((im_center, im_left, im_right))
        yield list


driving_log = pd.read_csv('training_data/driving_log.csv')
nb_classes = len(driving_log)

print ("features: ", nb_classes)

x = tf.placeholder(tf.float32, (None, 160, 320, 3))
# resized = tf.image.resize_images(x, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer specifically for
# traffic signs model.
fc7 = AlexNet(x, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
# im1 = imread("construction.jpg").astype(np.float32)
# im1 = im1 - np.mean(im1)
# im2 = imread("stop.jpg").astype(np.float32)
# im2 = im2 - np.mean(im2)

image_generator = generate_image()
car_images = []
for i in range(nb_classes):
    car_images = (next(image_generator))

# print(images)
# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: car_images})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (driving_log.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))


