import rospy
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,vehicle_mass,wheel_radius, wheel_base, steer_ratio, min_speed,
        max_lat_accel, max_steer_angle):
        self.vehicle_mass=vehicle_mass
        self.wheel_radius=wheel_radius
        self.yaw_controller = YawController(wheel_base,steer_ratio,min_speed,max_lat_accel,max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        minimum_throttle = 0.0
        maximum_throttle = 0.2

        self.throttle_controller = PID(kp,ki,kd,minimum_throttle,maximum_throttle)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.

class PID(object):
    def __init__(self,kp,ki)