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
        current_velocity_unfiltered,
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

        #
        # Filter velocity
        current_velocity = self.velocity_filter.filt(current_velocity_unfiltered)

        #
        # Determine steering
        steer = self.yaw_controller.get_steering(
            target_linear_velocity,
            target_angular_velocity,
            current_velocity,
        )

        #
        # Figure out throttle/braking
        throttle = 0.
        brake = 0.

        current_time = rospy.Time.now()

        if self.last_time != None:
            # Find throttle
            dt = (current_time - self.last_time).nsecs / 1e9
            velocity_error = target_linear_velocity - current_velocity
            throttle = self.throttle_controller.step(velocity_error, dt)

            # Braking
            if (  # STAY
                # want to stop
                target_linear_velocity == 0. and
                # currently stopped
                current_velocity < 0.1
            ):
                throttle = 0.
                brake = 700  # Newton meters to hold Carla in place
            elif (  # SLOW
                # throttle is low
                throttle < 0.1 and
                # need to slow down
                target_linear_velocity < current_velocity
            ):
                throttle = 0
                brake = abs(max(velocity_error, self.decel_limit)) * self.vehicle_mass * self.wheel_radius

        self.last_time = current_time

        return throttle, brake, steer
