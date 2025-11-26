#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geographic_msgs.msg import GeoPoseStamped
from geometry_msgs.msg import PoseStamped
from threading import Lock
import csv

PUBLISH_RATE_HZ = 20.0  # Hz

class NavigationNodeLatencyOptimized(Node):
    def __init__(self):
        super().__init__('navigation_node_latency_optimized')

        # QoS for MAVROS topics
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self.create_subscription(GeoPoseStamped, '/next_gps_waypoint', self.waypoint_cb, qos_profile=mavros_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.px4_pose_cb, qos_profile=mavros_qos)
        # Optional: add GPS/altitude subscriptions if needed

        # Publisher
        self.setpoint_pub = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)

        # Internal state
        self.current_wp = None
        self.wp_lock = Lock()

        # CSV logger
        self.csv = open('latency_navigation_optimized.csv', 'w', newline='')
        self.csvw = csv.writer(self.csv)
        self.csvw.writerow([
            'waypoint_id', 'wp_publish_ns', 'wp_received_ns', 'wp_received_latency_ms',
            'px4_response_ns', 'wp_to_px4_ms', 'total_end_to_end_ms'
        ])

        # Timer for continuous publishing (PX4 offboard)
        timer_period = 1.0 / PUBLISH_RATE_HZ
        self.create_timer(timer_period, self.publish_loop)

        self.get_logger().info("✅ navigation_node_latency_optimized started")

    def waypoint_cb(self, msg: GeoPoseStamped):
        recv_ns = self.get_clock().now().nanoseconds
        # Reconstruct publish time from header
        pub_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Use pose.orientation.z as a simple waypoint ID (example)
        wp_id = int(getattr(msg.pose.orientation, 'z', 0))
        # Use pose.orientation.y to store inference start timestamp (optional)
        inference_start_ns = int(getattr(msg.pose.orientation, 'y', 0))

        recv_latency_ms = (recv_ns - pub_ns) / 1e6

        # Store waypoint for PX4 response correlation
        with self.wp_lock:
            self.current_wp = {
                'id': wp_id,
                'pub_ns': pub_ns,
                'recv_ns': recv_ns,
                'inference_start_ns': inference_start_ns
            }

        # Log reception latency (PX4 fields empty for now)
        self.csvw.writerow([wp_id, pub_ns, recv_ns, f"{recv_latency_ms:.2f}", "", "", ""])
        self.csv.flush()

        self.get_logger().info(f"[NAV] WP {wp_id} recv latency {recv_latency_ms:.2f} ms")

    def px4_pose_cb(self, msg: PoseStamped):
        with self.wp_lock:
            if self.current_wp is None:
                return

            wp = self.current_wp
            px4_ns = self.get_clock().now().nanoseconds
            wp_to_px4_ms = (px4_ns - wp['pub_ns']) / 1e6

            # Total end-to-end latency (AI → PX4)
            total_ms = (px4_ns - wp['inference_start_ns']) / 1e6 if wp['inference_start_ns'] else None
            total_ms_str = f"{total_ms:.2f}" if total_ms is not None else "N/A"

            # Append row to CSV
            self.csvw.writerow([
                wp['id'], wp['pub_ns'], wp['recv_ns'],
                f"{(wp['recv_ns'] - wp['pub_ns'])/1e6:.2f}",
                px4_ns, f"{wp_to_px4_ms:.2f}", total_ms_str
            ])
            self.csv.flush()

            self.get_logger().info(f"[E2E] WP {wp['id']} wp->px4 {wp_to_px4_ms:.2f} ms total {total_ms_str} ms")

            # Clear current waypoint for next tracking
            self.current_wp = None

    def publish_loop(self):
        # Re-publish the last waypoint continuously for PX4 offboard control
        with self.wp_lock:
            wp = self.current_wp

        if wp:
            # For demonstration: create a GeoPoseStamped and publish
            msg = GeoPoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            # TODO: Add actual lat/lon/alt stored in wp if needed
            # msg.pose.position.latitude = wp['lat']
            # msg.pose.position.longitude = wp['lon']
            # msg.pose.position.altitude = wp['alt']
            # self.setpoint_pub.publish(msg)
            pass  # Replace with actual publishing logic if you store lat/lon

    def destroy_node(self):
        self.csv.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNodeLatencyOptimized()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
