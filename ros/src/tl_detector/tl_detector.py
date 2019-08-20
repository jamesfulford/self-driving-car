#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

WAYPOINT_VISIBILITY_HORIZON = 100
"""
A light is considered "visible" when its stopline is
under WAYPOINT_VISIBILITY_HORIZON waypoints ahead of
the car's current position
"""

SAMPLE_RATE = 10

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose_initialized = False
        self.pose = None

        self.waypoints_initialized = False
        self.waypoints = None
        self.waypoints_tree = None
        self.stop_line_waypoint_indices = []

        self.lights_initialized = False
        self.lights = []
        self.lights_tree = None

        self.camera_image_initialized = False
        self.camera_image = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)


        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        rospy.loginfo('Initializing')
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # # For taking pictures
        # self.clicker = 1451

        rospy.loginfo('Initialized')

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

        self.pose_initialized = True

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.waypoints_tree = KDTree([
            [
                waypoint.pose.pose.position.x,
                waypoint.pose.pose.position.y
            ] for waypoint in self.waypoints
        ])

        self.stop_line_waypoint_indices = [
            self.get_closest_waypoint(l[0], l[1]) for l in self.config['stop_line_positions']
        ]

        self.waypoints_initialized = True

    def traffic_cb(self, msg):
        self.lights = msg.lights
        self.lights_tree = KDTree([
            [
                l.pose.pose.position.x,
                l.pose.pose.position.y,
            ] for l in self.lights
        ])
        self.lights_initialized = True

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = msg
        self.camera_image_initialized = True
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x, y (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoints_tree.query([x, y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if not self.camera_image_initialized:
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # self.clicker += 1
        # if self.clicker % SAMPLE_RATE == 0:
        #     rospy.loginfo("click! {}".format(light.state))
        #     cv2.imwrite("/capstone/data/image_data/{}/{}.png".format(light.state, self.clicker // SAMPLE_RATE), cv_image)

        # return light.state  # for now

        # Get classification
        guess = self.light_classifier.get_classification(cv_image)
        rospy.loginfo("guess: {}, actual: {}".format(guess, light.state))
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (self.pose_initialized and self.waypoints_initialized and self.lights_initialized):
            car_position_i = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            next_stoplight_waypoint_i = (
                car_position_i + min(
                    map(
                        lambda l: (l - car_position_i) % len(self.waypoints),
                        self.stop_line_waypoint_indices,
                    )
                )
            ) % len(self.waypoints)

            light_i = self.lights_tree.query([
                self.waypoints[next_stoplight_waypoint_i].pose.pose.position.x,
                self.waypoints[next_stoplight_waypoint_i].pose.pose.position.y,
            ])[1]
            light = self.lights[light_i]

            rospy.loginfo("checkpoint --[{})---{}|  {}----".format(
                car_position_i,
                next_stoplight_waypoint_i,
                self.get_closest_waypoint(light.pose.pose.position.x, light.pose.pose.position.y)
            ))

            # TODO(james.fulford): Refine
            if ((next_stoplight_waypoint_i - car_position_i) % len(self.waypoints) < WAYPOINT_VISIBILITY_HORIZON):
                state = self.get_light_state(light)
                return next_stoplight_waypoint_i, state
            else:
                return next_stoplight_waypoint_i, TrafficLight.UNKNOWN

        else:
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
