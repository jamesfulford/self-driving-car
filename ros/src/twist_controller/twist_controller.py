import rospy

from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(
        self,
        vehicle_mass,
        fuel_capacity,
        brake_deadband,
        decel_limit,
        accel_limit,
        wheel_radius,
        wheel_base,
        steer_ratio,
        max_lat_accel,
        max_steer_angle,
        min_speed,
    ):
        # Control Yaw
        self.yaw_controller = YawController(
            wheel_base,
            steer_ratio,
            min_speed,
            max_lat_accel,
            max_steer_angle
        )

        # Control Throttle
        kp = 0.15
        ki = 0.0
        kd = 0.09
        self.throttle_controller = PID(kp, ki, kd, mn=0.0, mx=0.2)

        # Control Velocity
        tau = 12
        ts = 1
        self.velocity_filter = LowPassFilter(tau, ts)

        self.last_time = None

        # Save just in case
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

    def control(
        self,
        current_velocity,
        target_linear_velocity,
        target_angular_velocity,
        dbw_enabled,
    ):
        #
        # Disabled case
        #
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        #
        # Main controller code
        #
        steer = self.yaw_controller.get_steering(
            target_linear_velocity,
            target_angular_velocity,
            current_velocity,
        )

        throttle = 0.
        brake = 0.

        current_time = rospy.Time.now()

        if self.last_time != None:
            dt = (current_time - self.last_time).nsecs / 1e9
            cte = target_linear_velocity - current_velocity
            acceleration = self.throttle_controller.step(cte, dt)

            self.velocity_filter.filt(acceleration)
            if self.velocity_filter.ready:
                acceleration = self.velocity_filter.get()

            if acceleration > 0:
                throttle = acceleration
            else:
                brake = self.vehicle_mass * abs(acceleration) * self.wheel_radius

        self.last_time = current_time

        return throttle, brake, steer
