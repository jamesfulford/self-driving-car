# from styx_msgs.msg import TrafficLight
from keras.models import load_model
import tensorflow as tf
import h5py
import numpy as np


ONEHOT_INDEX_TO_NUMBER = [0, 1, 2]


def onehot_to_number(onehot):
    return ONEHOT_INDEX_TO_NUMBER[np.argmax(onehot)]


class TLClassifier(object):
    def __init__(self):
        self.model = load_model("/capstone/data/model.h5")
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        inp = np.array([image])
        # print inp.shape

        status = 0
        with self.graph.as_default():
            guess = self.model.predict(inp, batch_size=1)
            # print guess[0]
            status = onehot_to_number(guess[0])

        return status

if __name__ == '__main__':
    from glob import glob
    import cv2

    cl = TLClassifier()

    for path in glob("/capstone/data/image_data/*/*.png"):
        img = cv2.imread(path)
        status = cl.get_classification(img)
        print path, status
