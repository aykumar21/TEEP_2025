#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from geographic_msgs.msg import GeoPoseStamped
import torch
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage
from model import UNet
import cv2
import time
import csv

class CameraNodeLatencyBaseline(Node):
    def __init__(self):
        super().__init__('camera_node_latency_baseline')
        self.bridge = CvBridge()

        # ✅ MAVROS QoS fix — must match PX4 publishers (BEST_EFFORT)
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ✅ Image topic QoS — same as camera driver
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.listener_callback, qos_profile=image_qos)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile=mavros_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_profile=mavros_qos)

        # Publishers
        self.waypoint_pub = self.create_publisher(GeoPoseStamped, '/next_gps_waypoint', 10)

        # State
        self.current_gps = None
        self.drone_altitude = 3.0
        self.waiting_for_gps_logged = False  # Prevent repeated logs

        # Load U-Net model
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load('unet_model.pth', map_location='cpu'))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # CSV logging
        self.csv_file = open('camera_node_latency.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Waypoint_Timestamp_ns', 'AI_Inference_Latency_ms'])

        self.get_logger().info("✅ Camera node for latency baseline started successfully.")

    def gps_callback(self, msg):
        self.current_gps = (msg.latitude, msg.longitude)

    def local_pose_callback(self, msg):
        self.drone_altitude = msg.pose.position.z

    def listener_callback(self, msg):
        if self.current_gps is None:
            if not self.waiting_for_gps_logged:
                self.get_logger().warn("Waiting for GPS fix...")
                self.waiting_for_gps_logged = True
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)

        # Measure AI inference latency
        start_inference = time.time()
        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
        end_inference = time.time()
        ai_latency_ms = (end_inference - start_inference) * 1000

        # Publish waypoint
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
    node = CameraNodeLatencyBaseline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
