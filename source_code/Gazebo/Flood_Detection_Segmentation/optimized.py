#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from cv_bridge import CvBridge
import torch
from torchvision import transforms
import numpy as np
from model import UNet
import time
import csv
import cv2

class CameraNodeLatencyOptimized(Node):
    def __init__(self):
        super().__init__('camera_node_latency_optimized')

        # Callback group for async execution
        self.callback_group = ReentrantCallbackGroup()

        # CvBridge for ROS Image → OpenCV
        self.bridge = CvBridge()

        # Determine device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # QoS Profiles
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile=image_qos,
                                 callback_group=self.callback_group)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile=mavros_qos,
                                 callback_group=self.callback_group)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback,
                                 qos_profile=mavros_qos, callback_group=self.callback_group)

        # Publisher
        self.waypoint_pub = self.create_publisher(GeoPoseStamped, '/next_gps_waypoint', 10)

        # State
        self.current_gps = None
        self.drone_altitude = 3.0
        self.waiting_for_gps_logged = False

        # Load model
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('unet_model.pth', map_location=self.device))
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

        # CSV logging
        self.csv_file = open('camera_node_latency_optimized.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Waypoint_Timestamp_ns', 'AI_Inference_Latency_ms'])

        self.get_logger().info("✅ Camera node latency optimized started.")

    def gps_callback(self, msg: NavSatFix):
        self.current_gps = (msg.latitude, msg.longitude)

    def local_pose_callback(self, msg: PoseStamped):
        self.drone_altitude = msg.pose.position.z

    def image_callback(self, msg: Image):
        if self.current_gps is None:
            if not self.waiting_for_gps_logged:
                self.get_logger().warn("Waiting for GPS fix...")
                self.waiting_for_gps_logged = True
            return

        # Convert ROS Image → OpenCV → RGB tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256))

        # AI inference
        start_time = time.time()
        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
        end_time = time.time()
        ai_latency_ms = (end_time - start_time) * 1000

        # Publish waypoint immediately
        now = self.get_clock().now()
        waypoint_msg = GeoPoseStamped()
        waypoint_msg.header.frame_id = "map"
        waypoint_msg.header.stamp = now.to_msg()
        waypoint_msg.pose.position.latitude = 28.5445
        waypoint_msg.pose.position.longitude = 77.2721
        waypoint_msg.pose.position.altitude = 10.0
        self.waypoint_pub.publish(waypoint_msg)

        # Log latency
        self.csv_writer.writerow([now.nanoseconds, f"{ai_latency_ms:.2f}"])
        self.get_logger().info(f"[AI] Inference latency: {ai_latency_ms:.2f} ms | Waypoint timestamp: {now.nanoseconds}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNodeLatencyOptimized()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
