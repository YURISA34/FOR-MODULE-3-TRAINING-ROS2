#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCRNode(Node):
    def __init__(self):
        super().__init__('ocr_node')

        # Publisher for recognized text
        self.publisher_ = self.create_publisher(String, 'ocr_text', 10)

        # Subscribe to camera topic (sensor QoS, no cv_bridge needed)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Change to your camera topic
            self.image_callback,
            qos_profile_sensor_data)

        # RapidOCR: ONNXRuntime CPU backend, ~15MB models, no PyTorch,
        # no GPU, and no forced numpy upgrade (numpy<2.0,>=1.19.5)
        self.engine = RapidOCR()
        self.conf_threshold = 0.8

        self.get_logger().info('RapidOCR Node has started (CPU-only).')

    @staticmethod
    def _yuy2_to_rgb(data, height, width):
        """Convert packed YUV 4:2:2 (YUYV/YUY2) to RGB using pure numpy.
        Many USB/V4L2 cameras default to this format. No cv2 or cv_bridge
        needed here, so this won't touch your pinned numpy==1.21.5."""
        yuyv = np.frombuffer(data, dtype=np.uint8).reshape(
            height, width // 2, 4).astype(np.int32)

        y0, u, y1, v = yuyv[..., 0], yuyv[..., 1], yuyv[..., 2], yuyv[..., 3]

        def _convert(y, u, v):
            c = y - 16
            d = u - 128
            e = v - 128
            r = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
            g = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
            b = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
            return r, g, b

        r0, g0, b0 = _convert(y0, u, v)
        r1, g1, b1 = _convert(y1, u, v)

        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, 0::2, 0], rgb[:, 0::2, 1], rgb[:, 0::2, 2] = r0, g0, b0
        rgb[:, 1::2, 0], rgb[:, 1::2, 1], rgb[:, 1::2, 2] = r1, g1, b1
        return rgb

    def image_callback(self, msg):
        try:
            if msg.encoding == 'yuv422_yuy2':
                img = self._yuy2_to_rgb(msg.data, msg.height, msg.width)
            elif msg.encoding in ('bgr8', 'rgb8'):
                # Manual decode instead of cv_bridge - same pattern as your
                # other perception nodes, keeps the pipeline numpy-safe
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                if msg.encoding == 'bgr8':
                    img = img[:, :, ::-1]  # BGR -> RGB
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return

            result, _ = self.engine(img)

            if result:
                for box, text, score in result:
                    if score > self.conf_threshold:
                        self.get_logger().info(f"Detected: {text} ({score:.2f})")
                        self.publisher_.publish(String(data=text))

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
