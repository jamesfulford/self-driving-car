#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the
`waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear
and angular velocities. You can subscribe to any other message that you find
important or refer to the document for list of messages subscribed to by the
reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller`
class is the status of `dbw_enabled`. While in the simulator, its enabled all
the time, in the real car, that will not be the case. This may cause your PID
controller to accumulate error because the car could temporarily be driven by
a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values
(like vehicle_mass, wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and
other utility classes. You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on
the various publishers that we have created in the `__init__` function.

'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # Inputs
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        min_speed = 0.1

        self.twist_cmd_sub = rospy.Subscriber(
            "/twist_cmd",
            TwistStamped,
            self.twist_cmd_handler
        )
        self.dbw_enabled_sub = rospy.Subscriber(
            "/vehicle/dbw_enabled",
            Bool,
            self.dbw_enabled_handler
        )
        self.current_velocity_sub = rospy.Subscriber(
            "/current_velocity",
            TwistStamped,
            self.current_velocity_handler
        )

        # State
        self.initialized = False
        self.dbw_enabled = False
        self.current_velocity = None
        self.twist_cmd = None
        self.controller = Controller(
            vehicle_mass=vehicle_mass,
            fuel_capacity=fuel_capacity,
            brake_deadband=brake_deadband,
            decel_limit=decel_limit,
            accel_limit=accel_limit,
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle,
            min_speed=min_speed,
        )

        # Outputs
        self.steering_cmd_pub = rospy.Publisher(
            '/vehicle/steering_cmd',
            SteeringCmd,
            queue_size=1
        )
        self.throttle_cmd_pub = rospy.Publisher(
            '/vehicle/throttle_cmd',
            ThrottleCmd,
            queue_size=1
        )
        self.brake_cmd_pub = rospy.Publisher(
            '/vehicle/brake_cmd',
            BrakeCmd,
            queue_size=1
        )

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            throttle, brake, steering = (0.0, 700.0, 0.0)

            initialized = bool(self.initialized and self.current_velocity and self.twist_cmd)
            if initialized:
                # If DBW is disabled, still register measurements
                # just in case it is re-enabled
                throttle, brake, steering = self.controller.control(
                    # Current velocity
                    self.current_velocity.twist.linear.x,
                    # Target linear velocity
                    self.twist_cmd.twist.linear.x,
                    # Target angular velocity
                    self.twist_cmd.twist.angular.z,
                    # DBW is enabled (if disabled, some controllers reset)
                    self.dbw_enabled,
                )

            if self.dbw_enabled:
                if initialized:
                    rospy.loginfo( "Controller: input {} {} {} {}".format(
                        # Current velocity
                        self.current_velocity.twist.linear.x,
                        # Target linear velocity
                        self.twist_cmd.twist.linear.x,
                        # Target angular velocity
                        self.twist_cmd.twist.angular.z,
                        # DBW is enabled (if disabled, some controllers reset)
                        self.dbw_enabled,
                    ))
                    rospy.loginfo("Controller: publishing {} {} {}".format(throttle, brake, steering))
                # will not publish if dbw_enabled is not initialized
                # or if drive by wire is disabled
                self.publish(throttle, brake, steering)

            rate.sleep()

    def twist_cmd_handler(self, twist_cmd):
        self.twist_cmd = twist_cmd

    def dbw_enabled_handler(self, dbw_enabled):
        self.dbw_enabled = dbw_enabled.data
        self.initialized = True

    def current_velocity_handler(self, current_velocity):
        self.current_velocity = current_velocity

    def publish(self, throttle, brake, steer):
        rospy.loginfo("dbw_node: publishing throttle {} brake {} steer {}".format(throttle, brake, steer))

        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_cmd_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steering_cmd_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_cmd_pub.publish(bcmd)


if __name__ == '__main__':
    print("Starting dbw node")
    DBWNode()
