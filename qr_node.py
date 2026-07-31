#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
from pyzbar import pyzbar
import numpy as np

class QRCodeDetector(Node):
    def __init__(self):
        super().__init__('qr_code_detector')

        # Subscribers
        self.sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )

        # Publishers
        self.text_pub = self.create_publisher(String, 'qr_text', 10)
        self.marker_pub = self.create_publisher(Marker, 'qr_marker', 10)

        # CV bridge
        self.bridge = CvBridge()

        self.get_logger().info("QR Code Detector started.")

    def image_callback(self, msg: Image):
        # Convert to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detect QR codes
        qrcodes = pyzbar.decode(frame)

        for qr in qrcodes:
            qr_data = qr.data.decode("utf-8")
            (x, y, w, h) = qr.rect

            # Publish text
            self.text_pub.publish(String(data=qr_data))
            self.get_logger().info(f"QR Code: {qr_data}")

            # Create marker in RViz
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id  # usually "camera_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "qr_codes"
            marker.id = hash(qr_data) % 1000
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Position in image plane (simple 2D projection → place at z=1.0m)
            marker.pose.position.x = (x + w/2) * 0.001  # scale to meters
            marker.pose.position.y = (y + h/2) * 0.001
            marker.pose.position.z = 1.0
            marker.pose.orientation.w = 1.0

            marker.scale.z = 0.2  # text size
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.text = qr_data

            self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = QRCodeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

