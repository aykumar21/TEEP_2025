#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import WaypointPush, WaypointClear
from mavros_msgs.msg import Waypoint
import paho.mqtt.client as mqtt
import time
import csv
import os

DEFAULT_ALT = 10.0
MISSION_FILE = "mission.waypoints"
LOG_FILE = "latencies.csv"


class MqttMissionUploader(Node):
    def __init__(self):
        super().__init__("mqtt_mission_uploader")

        # ROS services
        self.wp_clear = self.create_client(WaypointClear, "/mavros/mission/clear")
        self.wp_push = self.create_client(WaypointPush, "/mavros/mission/push")

        # waypoint buffer
        self.waypoints = []

        # Prepare latency CSV
        self.init_csv()

        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self.on_mqtt_msg
        self.mqtt_client.connect("172.20.10.2", 1883, 60)
        self.mqtt_client.subscribe("uav/flood_detection")
        self.mqtt_client.loop_start()

        self.get_logger().info("Connected → Subscribed to uav/flood_detection")

    # ----------------------------------------------------------
    # Initialize CSV logging
    def init_csv(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["jetson_ts", "recv_ts", "mqtt_latency",
                                 "px4_ack_ts", "push_latency", "total_latency"])

    # ----------------------------------------------------------
    # MQTT callback
    def on_mqtt_msg(self, client, userdata, message):
        msg = message.payload.decode()
        recv_ts = time.time()   # when Jetson message is received

        self.get_logger().info(f"MQTT Msg → {msg}")

        try:
            parts = msg.split(",")
            lat = float(parts[0])
            lon = float(parts[1])
            jetson_ts = float(parts[2])     # Sent timestamp from Jetson

            mqtt_latency = recv_ts - jetson_ts

            self.get_logger().info(f"Latency → MQTT: {mqtt_latency:.3f} sec")

            alt = float(DEFAULT_ALT)

            wp = Waypoint()
            wp.is_current = False
            wp.frame = 3
            wp.command = 16
            wp.autocontinue = True

            wp.param1 = 0.0
            wp.param2 = 0.0
            wp.param3 = 0.0
            wp.param4 = 0.0

            wp.x_lat = lat
            wp.y_long = lon
            wp.z_alt = alt

            self.waypoints.append(wp)

            self.write_wp_file()

            px4_ack_ts, push_latency = self.push_to_px4()

            total_latency = px4_ack_ts - jetson_ts

            self.get_logger().info(
                f"✅ LATENCY SUMMARY:\n"
                f"MQTT      = {mqtt_latency:.3f} sec\n"
                f"PX4 Push  = {push_latency:.3f} sec\n"
                f"TOTAL     = {total_latency:.3f} sec\n"
            )

            self.log_latency(jetson_ts, recv_ts, mqtt_latency,
                             px4_ack_ts, push_latency, total_latency)

        except Exception as e:
            self.get_logger().error(f"Error parsing MQTT message: {e}")

    # ----------------------------------------------------------
    def write_wp_file(self):
        with open(MISSION_FILE, "w") as f:
            f.write("QGC WPL 110\n")

            for i, wp in enumerate(self.waypoints):
                f.write(
                    f"{i}\t0\t3\t16\t0.000000\t0.000000\t0.000000\t0.000000\t"
                    f"{wp.x_lat:.7f}\t{wp.y_long:.7f}\t{wp.z_alt:.3f}\t1\n"
                )

        self.get_logger().info("✅ mission.waypoints updated")

    # ----------------------------------------------------------
    def push_to_px4(self):
        start = time.time()

        # Clear old mission
        while not self.wp_clear.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /mavros/mission/clear...")

        req = WaypointClear.Request()
        res = self.wp_clear.call(req)

        if res.success:
            self.get_logger().info("✅ Previous mission cleared")

        # Upload waypoints
        while not self.wp_push.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /mavros/mission/push...")

        req2 = WaypointPush.Request()
        req2.start_index = 0
        req2.waypoints = self.waypoints
        res2 = self.wp_push.call(req2)

        end = time.time()
        push_latency = end - start

        if res2.success:
            self.get_logger().info(f"✅ Mission uploaded → Count = {res2.wp_transfered}")
        else:
            self.get_logger().error("❌ Mission upload failed!")

        return end, push_latency

    # ----------------------------------------------------------
    def log_latency(self, jetson_ts, recv_ts, mqtt_latency,
                    px4_ack_ts, push_latency, total_latency):

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([jetson_ts, recv_ts, mqtt_latency,
                             px4_ack_ts, push_latency, total_latency])

        self.get_logger().info("✅ Latency logged")

    # ----------------------------------------------------------
def main():
    rclpy.init()
    node = MqttMissionUploader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
