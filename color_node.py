#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class RedDetector(Node):
    def __init__(self):
        super().__init__('red_detector')

        # Subscriber to camera image
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',   # change this topic to match your camera
            self.image_callback,
            10)

        # Publisher for annotated image
        self.publisher_ = self.create_publisher(Image, '/red/annotated', 10)

        # CV Bridge
        self.bridge = CvBridge()

        self.get_logger().info("Red Detector Node started.")

    def image_callback(self, msg: Image):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range (two ranges because red wraps around HSV hue)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Find contours of red regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # filter noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "RED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        annotated_msg.header = msg.header
        self.publisher_.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RedDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
