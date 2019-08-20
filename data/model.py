import glob
from math import ceil

import numpy as np
from scipy import ndimage
from PIL import Image
import cv2

from keras.models import Sequential
from keras.layers import (
    Activation,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)


RED =       [ 1, 0, 0 ]
YELLOW =    [ 0, 0, 1 ]  # our code treats yellow as green, so might as well
GREEN =     [ 0, 0, 1 ]
NUMBER_TO_1HOT = [ RED, YELLOW, GREEN ]  # See TrafficLight enum


#
# Configurations
#

# Assumes running in container
images_glob = "/capstone/data/image_data/*/*.png"
slash_index_of_class = -2  # should use  ^ this

input_shape = (600, 800, 3)

batch_size = 64

validation_set_size = .2

epochs = 10

#
# Get data from filesystem
#
samples = glob.glob(images_glob)
samples = filter(lambda p: "/1/" not in p, samples)

print("Samples: {}".format(len(samples)))

#
# Extract and mangle data
#
def get_data_from_sample(sample):
    image = cv2.cvtColor(ndimage.imread(sample), cv2.COLOR_RGB2BGR)
    p = Image.fromarray(image)

    value = NUMBER_TO_1HOT[
        int(sample.split('/')[slash_index_of_class])
    ]

    images = [
        image,
        np.fliplr(image),  # flipping left-to-right
        np.array(p.rotate(5, expand=False)),  # rotating
        np.array(p.rotate(365 - 5, expand=False)),  # rotating the other way
        cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),  # BGR -> grayscale -> "BGR"
    ]
    return (
        images,
        # Assumes value is invariant
        [value for _ in images]
    )

loaded_sample_packs = map(get_data_from_sample, samples)

features = []
for images in map(lambda sp: sp[0], loaded_sample_packs):
    features.extend(images)
features = np.array(features)

labels = []
for values in map(lambda sp: sp[1], loaded_sample_packs):
    labels.extend(values)
labels = np.array(labels)

print("features: {}, labels: {}".format(features.shape, labels.shape))

#
# Define model
#
model = Sequential()
model.add(Lambda(lambda i: (i - 128) / 256., input_shape=input_shape))  # Mean centering + normalization

model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(3))  # output layer
model.add(Activation("softmax"))


#
# Train model
#
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(

    features,
    labels,

    validation_split=validation_set_size,
    shuffle=True,

    epochs=epochs,
)
model.save("/capstone/data/model.h5")
