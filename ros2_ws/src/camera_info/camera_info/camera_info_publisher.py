import yaml

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage


class CameraInfoPublisher(Node):

    def __init__(self):
        super().__init__('camera_info_publisher')

        self.publisher_ = self.create_publisher(CameraInfo, '/camera_info', 10)
        self.subsription = self.create_subscription(
            CompressedImage,
            '/camera_left/image/compressed',
            self._image_callback,
            10
        )
        self.subsription

        self._calibration_yaml = f'/tmp/calibrationdata/ost.yaml'
        self._camera_info_msg = self._parse_yaml()

    def _image_callback(self, image_msg):
        self._camera_info_msg.header = image_msg.header
        self.publisher_.publish(self._camera_info_msg)

    def _parse_yaml(self):
        # Load data from file
        with open(self._calibration_yaml, 'r') as f:
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
