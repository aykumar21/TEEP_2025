#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from builtin_interfaces.msg import Time
import time

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        # Subscriber: raw camera feed from Gazebo
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',   # Gazebo camera topic
            self.image_callback,
            10
        )

        # Publisher: republished images with system time in header
        self.pub = self.create_publisher(Image, '/camera/image_system', 10)

        self.bridge = CvBridge()
        self.get_logger().info("CameraPublisher started, listening to /camera/image_raw")

    def image_callback(self, msg: Image):
        # Copy the incoming image
        img_msg = Image()
        img_msg.data = msg.data
        img_msg.height = msg.height
        img_msg.width = msg.width
        img_msg.encoding = msg.encoding
        img_msg.is_bigendian = msg.is_bigendian
        img_msg.step = msg.step

        # Overwrite timestamp with system (wall clock) time
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        img_msg.header.stamp = Time(sec=sec, nanosec=nanosec)
        img_msg.header.frame_id = "camera_system"

        # Publish
        self.pub.publish(img_msg)
        self.get_logger().debug("Republished image with system time")

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
