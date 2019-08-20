# from styx_msgs.msg import TrafficLight
from keras.models import load_model
import h5py
import numpy as np


ONEHOT_INDEX_TO_NUMBER = [0, 1, 2, 4]


def onehot_to_number(onehot):
    return ONEHOT_INDEX_TO_NUMBER[np.argmax(onehot)]


class TLClassifier(object):
    def __init__(self):
        self.model = load_model("/capstone/data/model.h5")

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        guess = self.model.predict(np.array([image]), batch_size=1)
        print guess, guess[0]
        return onehot_to_number(guess[0])

if __name__ == '__main__':
    from glob import glob
    import cv2

    cl = TLClassifier()

    for path in glob("/capstone/data/image_data/*/*.png"):
        img = cv2.imread(path)
        status = cl.get_classification(img)
        print path, status
