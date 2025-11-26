#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import time
import csv
import os
from datetime import datetime

class CameraLatencyNode(Node):
    def __init__(self):
        super().__init__('camera_latency_node')

        # Subscribe to system-timestamped images
        self.sub = self.create_subscription(
            Image,
            '/camera/image_system',   # listen to republished topic
            self.image_callback,
            10
        )

        # CSV log file
        self.log_file = os.path.join(
            os.path.dirname(__file__), 'camera_latency_log.csv'
        )
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "latency_ms"])

        self.get_logger().info(f"Logging latency to {self.log_file}")

    def image_callback(self, msg: Image):
        # Extract send time from header (system wall time)
        send_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Current wall time
        now = time.time()

        # Latency in ms
        latency_ms = (now - send_time) * 1000.0

        # Log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.get_logger().info(f"[{timestamp}] Image latency: {latency_ms:.2f} ms")

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, latency_ms])

def main(args=None):
    rclpy.init(args=args)
    node = CameraLatencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
