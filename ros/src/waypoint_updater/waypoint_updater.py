#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math
from copy import deepcopy

'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.

As mentioned in the doc, you should ideally first implement a version which
does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of
traffic lights too.

Please note that our simulator also provides the exact location of traffic
lights and their current status in `/vehicle/traffic_lights` message. You
can use this message to build this node as well as to verify your TL
classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# TODO(jafulfor): Extract to a ROS constant
LOOKAHEAD_WAYPOINTS = 100  # Number of waypoints to publish.

#
# James Fulford:
# My philosophy is simple: use very simple classes to manage state.
# Be strict and clear about what needs to be in state.
# Use functions for everything else.

# Bad code is often both complex and stateful code. Individually, these are easier to deal with.
# Stop passing parameters through mutating `self` or `this`.

def pythagorean_distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y) ** 2 + (a.z-b.z) ** 2)


def kmph2mps(velocity_kmph):
    return (velocity_kmph * 1000.) / (60. * 60.)


def waypoint_distances(waypoints):
    accumulated_distance = 0
    distances = [accumulated_distance]
    for i in range(1, len(waypoints)):
        accumulated_distance += pythagorean_distance(
            waypoints[i - 1].pose.pose.position,
            waypoints[i].pose.pose.position
        )
        distances.append(accumulated_distance)
    return distances


def use_linear_strategy():
    """
    Uses a linear function to generate target velocities.
    """
    max_velocity = kmph2mps(rospy.get_param("~velocity", 40))
    stop_line_buffer = 2.0

    def linear_strategy(distances_to_waypoints, current_velocity):
        # Target velocity function should be a line
        # going from (0, current_velocity)
        # to (last_waypoint - buffer, 0)
        # (after x-intercept, y = 0)
        d = max(distances_to_waypoints[-1] - stop_line_buffer, 0)  # stopping distance
        v = current_velocity  # amount by which to slow down within given distance

        # Protect against divide by 0 case
        if d < 0.01:
            return [0 for x in distances_to_waypoints]

        f = lambda x: min(
            max(

                # [0, d]: downward line:
                # y = (-v / d)x + v = (1 - (x/d)) * v
                (1. - (x / d)) * v,

                # (-inf, 0) && (d, +inf): flat
                # y = 0
                0
            ),
            # Never faster than maximum
            max_velocity
        )
        return map(f, distances_to_waypoints)

    return linear_strategy


class WaypointUpdater(object):
    """
    Stateful class surrounding waypoint_updater node

    Note: will not emit waypoints until camera is initialized.
    """
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoint_index_cb)
        rospy.Subscriber("/current_velocity", TwistStamped, self.current_velocity_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints',
            Lane,
            queue_size=1
        )

        # State by callback
        self.pose = None
        self.pose_initialized = False

        self.current_velocity = 0
        # self.velocity_filter = LowPassFilter(12, 1)
        self.current_velocity_initialized = False

        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.waypoints_initialized = False

        self.traffic_waypoint_index = None
        self.traffic_waypoint_index_initialized = False

        # Select strategy to use when braking for stoplight
        self.strategy = use_linear_strategy()

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.is_initialized():
                self.publish_waypoints(self.get_upcoming_waypoints())
            rate.sleep()

    def get_upcoming_waypoints(self):
        """
        Returns list of waypoints
        """
        start_index = self.get_next_waypoint_index()
        end_index = start_index + LOOKAHEAD_WAYPOINTS

        # Stop earlier for red light
        traffic_index = self.traffic_waypoint_index
        if traffic_index is not None:
            end_index = traffic_index

        waypoints = deepcopy(self.base_waypoints.waypoints[start_index:end_index])

        # Adjust waypoint target speeds if red light
        if traffic_index is not None and len(waypoints):
            distances_to_waypoints = waypoint_distances(waypoints)

            # Use strategy to decide how to slow down
            target_velocities = self.strategy(distances_to_waypoints, self.current_velocity)

            # Update waypoints
            for target_velocity, waypoint in zip(target_velocities, waypoints):
                waypoint.twist.twist.linear.x = target_velocity

        return waypoints

    def get_next_waypoint_index(self):
        position = [
            self.pose.pose.position.x,
            self.pose.pose.position.y
        ]
        i = self.waypoint_tree.query(position, 1)[1]

        # Find out closest versus next
        closest, prev = np.array(
            self.waypoints_2d[i]
        ), np.array(
            self.waypoints_2d[i - 1]
        )
        val = np.dot(closest - prev, np.array(position) - closest)
        # if (position - closest) is ahead of (closest - prev) vector
        # then closest waypoint is behind us and next waypoint is next
        return i if val <= 0 else (i + 1) % len(self.waypoints_2d)

    #
    # Inputs and initialization logic
    #
    def is_initialized(self):
        return (
            self.pose_initialized and
            self.waypoints_initialized and
            # All usage of traffic_waypoint_index is hidden behind a `is not None` check.
            # Enabling this line will block waypoint_updater from publishing until camera checkbox is enabled in simulator
            # self.traffic_waypoint_index_initialized and
            self.current_velocity_initialized
        )

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_initialized = True

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x
        self.current_velocity_initialized = True

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.waypoints_2d = [
            [
                waypoint.pose.pose.position.x,
                waypoint.pose.pose.position.y
            ] for waypoint in self.base_waypoints.waypoints
        ]
        self.waypoint_tree = KDTree(self.waypoints_2d)
        self.waypoints_initialized = True

    def traffic_waypoint_index_cb(self, waypoint_index):
        i = waypoint_index.data
        self.traffic_waypoint_index = i if i != -1 else None
        self.traffic_waypoint_index_initialized = True

    #
    # Outputs logic
    #
    def publish_waypoints(self, waypoints):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
