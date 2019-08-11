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
        # "Controller: yaw init 3 14.8 0.1 3.0 8.0"
        rospy.loginfo("Controller: yaw init {} {} {} {} {}".format(
            wheel_base,
            steer_ratio,
            min_speed,
            max_lat_accel,
            max_steer_angle
        ))

        # Control Throttle
        kp = 0.15
        ki = 0.0
        kd = 0.09
        self.throttle_controller = PID(kp, ki, kd, mn=0.0, mx=0.2)

        # Control Velocity
        tau = 12
        ts = 1
        self.velocity_filter = LowPassFilter(tau, ts)

        self.last_time = rospy.get_time()

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
        curr_velocity,
        linear_velocity,
        angular_velocity,
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
        current_velocity = self.velocity_filter.filt(curr_velocity)
        steer = self.yaw_controller.get_steering(
            linear_velocity,
            angular_velocity,
            current_velocity,
        )
        rospy.loginfo("Controller: Yaw: input: {} {} {}".format(
            linear_velocity,
            angular_velocity,
            current_velocity,
        ))

        v_error = linear_velocity - current_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(v_error, sample_time)
        brake = 0

        #
        # Carla-specific
        #
        if linear_velocity == 0.0 and current_velocity < 0.1:
            throttle = 0
            brake = 400  # Newton meters required for no movement
        elif throttle < 0.1 and v_error < 0:
            throttle = 0
            decel = max(v_error, self.decel_limit)
            # Torque
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steer
