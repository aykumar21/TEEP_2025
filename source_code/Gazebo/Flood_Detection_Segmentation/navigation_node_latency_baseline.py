#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import Altitude
from geometry_msgs.msg import PoseStamped
from collections import deque
import csv

class NavigationNodeLatencyBaseline(Node):
    def __init__(self, takeoff_altitude_target=5.0):
        super().__init__('navigation_node_latency_baseline')
        self.get_logger().info("✅ Autonomous GPS Navigation Node (Latency Baseline) Started")

        # ✅ MAVROS QoS (must match PX4 publisher)
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State variables
        self.gps_data = None
        self.amsl = None
        self.local_z = None
        self.takeoff_altitude_target = takeoff_altitude_target
        self.takeoff_mode_active = True
        self.takeoff_completed = False
        self.waypoint_queue = deque()
        self.current_waypoint = None

        # Subscribers
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos_profile=mavros_qos)
        self.create_subscription(Altitude, '/mavros/altitude', self.altitude_callback, qos_profile=mavros_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_profile=mavros_qos)
        self.create_subscription(GeoPoseStamped, '/next_gps_waypoint', self.waypoint_callback, qos_profile=mavros_qos)

        # Publisher for waypoint (simulated control)
        self.publisher = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)

        # CSV logger
        self.csv_file = open('navigation_node_latency.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Waypoint_Timestamp_ns', 'Waypoint_Received_Latency_ms'])

        self.get_logger().info("📝 Logging navigation latencies to 'navigation_node_latency.csv'")

    # ======== Callbacks ========

    def gps_callback(self, msg):
        self.gps_data = msg

    def altitude_callback(self, msg):
        self.amsl = msg.amsl

    def local_pose_callback(self, msg):
        self.local_z = msg.pose.position.z

    def waypoint_callback(self, msg):
        """Triggered when new GPS waypoint is received from camera node."""
        reception_time_ns = self.get_clock().now().nanoseconds

        # ✅ Fix: compute from sec + nanosec
        waypoint_publish_time_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Calculate latency
        nav_latency_ms = (reception_time_ns - waypoint_publish_time_ns) / 1e6

        # Log latency
        self.csv_writer.writerow([waypoint_publish_time_ns, f"{nav_latency_ms:.2f}"])
        self.get_logger().info(f"[NAV] Waypoint reception latency: {nav_latency_ms:.2f} ms")

        # Save waypoint
        lat = round(msg.pose.position.latitude, 6)
        lon = round(msg.pose.position.longitude, 6)
        alt = msg.pose.position.altitude
        self.waypoint_queue.append({"lat": lat, "lon": lon, "alt": alt})

        # Publish back to MAVROS to simulate control
        self.publisher.publish(msg)

    # ======== Cleanup ========

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()
        self.get_logger().info("📁 navigation_node_latency.csv saved and node shut down cleanly.")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNodeLatencyBaseline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
