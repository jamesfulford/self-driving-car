#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

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
LOOKAHEAD_WPS = 50  # Waypoints to publish. You can change this


def dl(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y) ** 2 + (a.z-b.z) ** 2)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # rospy.Subscriber('/traffic_waypoint', Waypoint, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints',
            Lane,
            queue_size=1
        )

        self.pose_initialized = False
        self.waypoints_initialized = False

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.is_initialized():
                self.publish_waypoints(self.get_next_waypoint_index())
            rate.sleep()

    def is_initialized(self):
        return self.pose_initialized and self.waypoints_initialized

    def publish_waypoints(self, idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[idx:idx + LOOKAHEAD_WPS]
        rospy.loginfo("next_waypoint: ({0.x}, {0.y}) [{1}]".format(lane.waypoints[0].pose.pose.position, idx))
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_initialized = True

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

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. Will implement later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, waypoint_1, waypoint_2):
        dist = 0
        wp1 = waypoint_1
        wp2 = waypoint_2
        for i in range(wp1, wp2 + 1):
            dist += dl(
                waypoints[wp1].pose.pose.position,
                waypoints[i].pose.pose.position
            )
            wp1 = i
        return dist

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
