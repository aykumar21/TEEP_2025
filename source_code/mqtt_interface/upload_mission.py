#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import WaypointPush
from mavros_msgs.msg import Waypoint

class MissionUploader(Node):
    def __init__(self):
        super().__init__("mission_uploader")

        self.cli = self.create_client(WaypointPush, "/mavros/mission/push")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /mavros/mission/push service...")

        mission_file = "mission.waypoints"
        self.get_logger().info(f"Loading mission file: {mission_file}")

        mission = self.parse_wp_file(mission_file)

        req = WaypointPush.Request()
        req.waypoints = mission

        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.done_callback)

    def parse_wp_file(self, filename):
        wps = []
        with open(filename, "r") as f:
            lines = f.readlines()

        # Skip first line: "QGC WPL xxx"
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue

            wp = Waypoint()

            # Convert properly
            wp.is_current = bool(int(parts[1]))       # <-- FIXED
            wp.frame      = int(parts[2])
            wp.command    = int(parts[3])

            wp.param1 = float(parts[4])
            wp.param2 = float(parts[5])
            wp.param3 = float(parts[6])
            wp.param4 = float(parts[7])

            wp.x_lat  = float(parts[8])
            wp.y_long = float(parts[9])
            wp.z_alt  = float(parts[10])

            wp.autocontinue = bool(int(parts[11])) if len(parts) > 11 else True   # <-- FIXED

            wps.append(wp)

        self.get_logger().info(f"Parsed {len(wps)} waypoints from mission file")
        return wps

    def done_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"✅ Mission upload successful! Count = {response.wp_transfered}")
            else:
                self.get_logger().error("❌ Mission upload FAILED!")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main():
    rclpy.init()
    node = MissionUploader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
