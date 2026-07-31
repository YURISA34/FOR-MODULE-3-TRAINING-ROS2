#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion
from std_msgs.msg import Header
from aruco_msgs.msg import Marker
import cv2
import cv2.aruco as aruco
import numpy as np
from visualization_msgs.msg import Marker as RVizMarker

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Publishers
        self.marker_pub = self.create_publisher(Marker, 'aruco_marker', 10)
        self.rviz_pub = self.create_publisher(RVizMarker, 'aruco_marker_viz', 10)

        # Subscribers
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)

        # CV bridge
        self.bridge = CvBridge()

        # Camera parameters (filled from /camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco dictionary (compatible getter)
        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        else:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        # Detector: old (<=4.6) vs new (>=4.7) API
        if hasattr(aruco, "DetectorParameters_create"):
            self.parameters = aruco.DetectorParameters_create()
            self._use_new_api = False
            self._detector = None
        else:
            self.parameters = aruco.DetectorParameters()
            self._use_new_api = True
            self._detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Whether OpenCV still has estimatePoseSingleMarkers
        self._has_estimate_pose = hasattr(aruco, "estimatePoseSingleMarkers")

        # Marker length in meters (adjust for your tags)
        self.marker_length = 0.05

        self.get_logger().info("Aruco detector node started.")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            # avoid spamming logs
            return

        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if self._use_new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is None or len(ids) == 0:
            return

        # Pose estimation: prefer OpenCV helper if available, else solvePnP fallback
        if self._has_estimate_pose:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
        else:
            rvecs, tvecs = self._estimate_pose_solvepnp_batch(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            # Build Pose
            pose = Pose()
            pose.position = Point(
                x=float(tvec[0][0]), y=float(tvec[0][1]), z=float(tvec[0][2])
            )
            rot_matrix, _ = cv2.Rodrigues(rvec)
            quat = self.rotation_matrix_to_quaternion(rot_matrix)
            pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

            # aruco_msgs/Marker
            marker_msg = Marker()
            marker_msg.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            marker_msg.id = int(marker_id)
            marker_msg.pose = PoseWithCovariance()
            marker_msg.pose.pose = pose
            marker_msg.confidence = 1.0

            # RViz marker (a thin square)
            vis = RVizMarker()
            vis.header.stamp = msg.header.stamp
            vis.header.frame_id = msg.header.frame_id
            vis.ns = "aruco_markers"
            vis.id = int(marker_id)
            vis.type = RVizMarker.CUBE
            vis.action = RVizMarker.ADD
            vis.pose = pose
            vis.scale.x = self.marker_length
            vis.scale.y = self.marker_length
            vis.scale.z = 0.01
            vis.color.r = 0.0
            vis.color.g = 1.0
            vis.color.b = 0.0
            vis.color.a = 0.8

            self.rviz_pub.publish(vis)
            self.marker_pub.publish(marker_msg)

    # ---------- helpers ----------

    def _estimate_pose_solvepnp_batch(self, corners_list, marker_length, K, dist):
        """
        Estimate pose for each detected marker using solvePnP.
        corners_list: list of (1,4,2) arrays in order TL, TR, BR, BL
        Returns rvecs, tvecs shaped like estimatePoseSingleMarkers output:
            rvecs: (N,1,3), tvecs: (N,1,3)
        """
        # Define 3D object points for a square marker centered at origin lying on Z=0
        # Corner ordering must match OpenCV's ArUco: TL, TR, BR, BL
        L = marker_length
        objp = np.array([
            [-L/2,  L/2, 0.0],  # TL
            [ L/2,  L/2, 0.0],  # TR
            [ L/2, -L/2, 0.0],  # BR
            [-L/2, -L/2, 0.0],  # BL
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for corners in corners_list:
            # corners shape: (1,4,2) -> (4,2)
            imgp = corners.reshape(4, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                # Fallback to iterative if IPPE fails (rare)
                success, rvec, tvec = cv2.solvePnP(
                    objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
            # shape to (1,3) for compatibility
            rvecs.append(rvec.reshape(1, 3))
            tvecs.append(tvec.reshape(1, 3))

        return np.array(rvecs, dtype=np.float64), np.array(tvecs, dtype=np.float64)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]."""
        q = np.empty((4,), dtype=np.float64)
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2, 1] - R[1, 2]) * s
            q[1] = (R[0, 2] - R[2, 0]) * s
            q[2] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) * 1.0 / s
                q[2] = (R[0, 2] + R[2, 0]) * 1.0 / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s
        return q

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
