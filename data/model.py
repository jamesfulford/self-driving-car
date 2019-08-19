import glob
from math import ceil

import numpy as np
from scipy import ndimage
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
import cv2

from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)

#
# Configurations
#

# Assumes running in container
images_glob = "/data/image_data/*/*.png"
slash_index_of_classification = -2

input_shape = (600, 800, 3)

batch_size = 8

validation_set_size = .2

epochs = 5

#
# Get data from filesystem
#
samples = glob.glob(images_glob)

print("Samples: {}".format(len(samples)))

train_samples, validation_samples = train_test_split(
    samples,
    test_size=validation_set_size
)


#
# Define generators to access data
#
def get_data_from_sample(sample):
    image = ndimage.imread(sample)
    p = Image.fromarray(image)

    value = int(sample.split('/')[slash_index_of_classification])

    images = [
        image,
        np.fliplr(image),  # flipping left-to-right
        np.array(p.rotate(15, expand=False)),  # rotating
        np.array(p.rotate(365 - 15, expand=False)),  # rotating the other way
        cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB),  # rgb -> grayscale -> "rgb"
    ]
    return (
        images,
        # Assumes value is invariant
        [value for _ in images]
    )


def data_generator(samples, batch_size=128):
    n = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for i in range(0, n, batch_size):
            batch_samples = samples[i:i + batch_size]

            batch_images, batch_values = [], []

            for sample in batch_samples:
                images, values = get_data_from_sample(sample)
                batch_images.extend(images)
                batch_values.extend(values)

            yield sklearn.utils.shuffle(
                np.array(batch_images),
                np.array(batch_values),
            )

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
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))  # output layer

#
# Train model
#
model.compile(loss="mse", optimizer="adam")
model.fit_generator(
    data_generator(
        train_samples,
        batch_size=batch_size,
    ),
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=data_generator(
        validation_samples,
        batch_size=batch_size,
    ),
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=epochs,
)
model.save("model.h5")
