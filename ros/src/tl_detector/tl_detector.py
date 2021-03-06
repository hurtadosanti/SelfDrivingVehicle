#!/usr/bin/env python
import os

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy import spatial
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

TRAFFIC_LIGHT_NAME_MAP = {
    TrafficLight.RED: 'RED',
    TrafficLight.YELLOW: 'YELLOW',
    TrafficLight.GREEN: 'GREEN',
    TrafficLight.UNKNOWN: 'UNKNOWN',
}

CLASSIFY_EVERY_N_IMAGES = 4


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        rospy.logdebug('__init__')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.image_counter = 0
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

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

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config['is_site']

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        rospy.loginfo('Current directory: %s', os.getcwd())

        # Get classifier model
        model_file = self.config['tensorflow_model']
        self.use_model = self.config.get('use_model', True)
        if os.path.exists(model_file) and self.use_model:
            rospy.loginfo('Using TensorFlow model: %s', model_file)
            self.light_classifier = TLClassifier(model_file)
        else:
            self.use_model = False

        self.listener = tf.TransformListener()


        rospy.spin()

    def pose_cb(self, msg):
        rospy.logdebug('pose_cb')
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [
                [wp.pose.pose.position.x, wp.pose.pose.position.y]
                for wp in waypoints.waypoints]
            self.waypoint_tree = spatial.KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        rospy.logdebug('traffic_cb')
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        rospy.logdebug('image_cb')
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:  # state of the light has changed
            self.state_count = 0
            self.state = state
            if state == TrafficLight.RED:
                self.last_wp = light_wp
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            # We just care if the light is read, for now
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        rospy.logdebug('Publishing %s to /traffic_waypoint', self.last_wp)
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        rospy.logdebug('get_closest_waypoint')
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.has_image:
            self.image_counter += 1
            rospy.logdebug('image_counter: %s', self.image_counter)
            if self.use_model and self.image_counter == CLASSIFY_EVERY_N_IMAGES:
                rospy.logdebug('Run prediction.')
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                light_state = self.light_classifier.get_classification(cv_image)
                self.image_counter = 0
            elif not self.is_site:
                light_state = light.state
            else:
                light_state = TrafficLight.UNKNOWN
        else:
            light_state = TrafficLight.UNKNOWN

        rospy.loginfo(
            'get_light_state(): Detected %s',
            TRAFFIC_LIGHT_NAME_MAP[light_state])
        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Store closest traffic light
        # Each traffic light comes with a line
        rospy.logdebug('process_traffic_lights')
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            # car_position = self.get_closest_waypoint(self.pose.pose)
            car_wp_idx = self.get_closest_waypoint(
                self.pose.pose.position.x, self.pose.pose.position.y)

            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        # self.waypoints = None
        # Note
        # TrafficLight.UNKNOWN -> keep the car moving for now
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
