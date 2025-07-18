import yaml

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage


class CameraInfoPublisher(Node):

    def __init__(self):
        super().__init__('camera_info_publisher')
        self.__calibration_yaml = self.declare_parameter(
            'yaml_path', 'calibrationdata/ost.yaml').value
        self.__image_topic = self.declare_parameter(
            'image_topic', 'camera_left/image/compressed').value
        self.__camera_info_topic = self.declare_parameter(
            'camera_info_topic', '/camera_info').value

        self.__publisher = self.create_publisher(
            CameraInfo, self.__camera_info_topic, 10)
        self.__subsription = self.create_subscription(
            CompressedImage,
            self.__image_topic,
            self.__image_callback,
            10
        )
        self.__subsription

        self.__camera_info_msg = self.__parse_yaml()

    def __image_callback(self, image_msg):
        self.__camera_info_msg.header = image_msg.header
        self.__publisher.publish(self.__camera_info_msg)

    def __parse_yaml(self):
        # Load data from file
        with open(self.__calibration_yaml, 'r') as f:
            calibration_data = yaml.safe_load(f)

        # Parse
        msg = CameraInfo()
        msg.width = calibration_data['image_width']
        msg.height = calibration_data['image_height']
        msg.distortion_model = calibration_data['distortion_model']
        msg.k = calibration_data['camera_matrix']['data']
        msg.d = calibration_data['distortion_coefficients']['data']
        msg.r = calibration_data['rectification_matrix']['data']
        msg.p = calibration_data['projection_matrix']['data']

        return msg


def main():
    rclpy.init()

    camera_info_publisher = CameraInfoPublisher()

    rclpy.spin(camera_info_publisher)

    camera_info_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
