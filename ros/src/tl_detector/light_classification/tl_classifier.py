from styx_msgs.msg import TrafficLight

import cv2

import prediction


class TLClassifier(object):

    def __init__(self, model_file):
        self.model = prediction.FrozenModel(model_file)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_class = self.model.predict(image_rgb)
        if output_class == prediction.TRAFFIC_LIGHT_STATE_RED:
            ret = TrafficLight.RED

        elif output_class == prediction.TRAFFIC_LIGHT_STATE_YELLOW:
            ret = TrafficLight.YELLOW

        elif output_class == prediction.TRAFFIC_LIGHT_STATE_GREEN:
            ret = TrafficLight.GREEN

        else:
            ret = TrafficLight.UNKNOWN

        return ret

