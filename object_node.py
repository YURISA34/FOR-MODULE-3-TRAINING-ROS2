#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import Pose2D, Point2D 
from std_msgs.msg import String
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
import cv2
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Subscriptions
        self.create_subscription(Image, "/image_raw", self.image_callback, 1)

        # Publishers — each on its own topic name. Two different message
        # types sharing one topic name is what crashed the original node.
        self.det_pub = self.create_publisher(Detection2DArray, "yolo_detections", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "yolo_markers", 10)
        self.text_pub = self.create_publisher(String, "yolo_labels", 10)
        self.image_pub = self.create_publisher(Image, "/yolo/annotated", 10)

        self.bridge = CvBridge()

        # YOLOv8n (nano): much lighter than yolov5s on CPU-only hardware.
        # ultralytics downloads the weights itself on first run (needs
        # internet once) — no torch.hub git clone involved.
        # Swap to 'yolo11n.pt' if you want the newer architecture; same
        # nano-class size and speed.
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info("YOLOv8n model loaded.")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # imgsz=320 keeps this responsive on a CPU-only laptop. Raise it
        # (e.g. 480, 640) if you have inference time to spare and want
        # better accuracy on small/far objects.
        results = self.model(frame, imgsz=320, verbose=False)[0]

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        markers = MarkerArray()
        labels_seen = []

        annotated_frame = frame.copy()

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            labels_seen.append(label)

            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )
            cv2.putText(
                annotated_frame,
                f"{label} ({conf:.2f})",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # Detection2D entry
            detection = Detection2D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            # vision_msgs 4.x nests id/score under .hypothesis — your apt
            # vision_msgs is 4.x, not the older flat .id/.score layout.
            hypothesis.hypothesis.class_id = str(cls)
            hypothesis.hypothesis.score = conf
            detection.results.append(hypothesis)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.position = Point2D()
            bbox.center.position.x = cx
            bbox.center.position.y = cy
            bbox.center.theta = 0.0
            bbox.size_x = x2 - x1
            bbox.size_y = y2 - y1
            detection.bbox = bbox

            detections_msg.detections.append(detection)

            # RViz marker
            marker = Marker()
            marker.header = msg.header
            marker.ns = "yolo"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.5  # fake depth (since 2D only)
            marker.pose.position.y = (x1 + x2) / 200.0
            marker.pose.position.z = (y1 + y2) / 200.0
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = f"{label} ({conf:.2f})"
            markers.markers.append(marker)

        # Publish
        image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        image_msg.header = msg.header
        self.image_pub.publish(image_msg)
        self.det_pub.publish(detections_msg)
        self.marker_pub.publish(markers)

        text_msg = String()
        text_msg.data = ", ".join(labels_seen)
        self.text_pub.publish(text_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()