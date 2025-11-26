#!/usr/bin/env python3
import paho.mqtt.client as mqtt
from pymavlink import mavutil
import time
import sys
import os
import csv

# ==============================
# CONFIGURATION
# ==============================
PX4_IP = "127.0.0.1"          # or 192.168.x.x if external PX4 device
PX4_PORT = 14550              # MAVLink port (default 14550 or 14560)
MQTT_BROKER = "192.168.1.93"  # Jetson’s IP (MQTT broker)
MQTT_TOPIC = "uav/flood_detection"
ALTITUDE = 10                 # Default waypoint altitude (m)
WAYPOINT_FILE = "mission.waypoints"
LATENCY_LOG = "e2e_latency.csv"
# ==============================


# ------------------------------
# Connect to PX4
# ------------------------------
def connect_px4():
    print(f"🔗 Connecting to PX4 SITL at {PX4_IP}:{PX4_PORT} ...")
    master = None
    for i in range(10):
        try:
            master = mavutil.mavlink_connection(f'udpout:{PX4_IP}:{PX4_PORT}')
            master.wait_heartbeat(timeout=10)
            print(f"✅ Connected to PX4 (System ID: {master.target_system})")
            return master
        except Exception as e:
            print(f"⚠️ Connection attempt {i+1} failed: {e}")
            time.sleep(2)
    print("❌ Could not connect to PX4.")
    sys.exit(1)


# ------------------------------
# Save waypoint locally (.waypoints format)
# ------------------------------
def save_waypoint_to_file(lat, lon, alt):
    if not os.path.exists(WAYPOINT_FILE):
        with open(WAYPOINT_FILE, "w") as f:
            f.write("QGC WPL 110\n")

    with open(WAYPOINT_FILE, "r") as f:
        lines = f.readlines()
    seq = len(lines) - 1

    with open(WAYPOINT_FILE, "a") as f:
        f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{alt}\t1\n")

    print(f"💾 Saved waypoint #{seq}: {lat:.6f}, {lon:.6f}, alt={alt} m")


# ------------------------------
# Upload mission & measure ACK latency
# ------------------------------
def upload_mission_to_px4(master, sent_ts, lat, lon, alt):
    if not os.path.exists(WAYPOINT_FILE):
        print("⚠️ No mission file found.")
        return

    with open(WAYPOINT_FILE, "r") as f:
        lines = f.readlines()[1:]  # skip header

    mission_items = []
    for seq, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) >= 11:
            lat_i, lon_i, alt_i = float(parts[8]), float(parts[9]), float(parts[10])
            mission_items.append((seq, lat_i, lon_i, alt_i))

    if not mission_items:
        print("⚠️ No valid waypoints found.")
        return

    print(f"⬆️ Uploading {len(mission_items)} waypoint(s) to PX4 ...")
    master.mav.mission_count_send(master.target_system, master.target_component, len(mission_items))

    # Handle requests for each waypoint
    for seq, lat_i, lon_i, alt_i in mission_items:
        req = master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=5)
        if not req:
            print(f"⚠️ Timeout waiting for request for waypoint #{seq}")
        else:
            pass  # can check req.seq == seq if needed

        master.mav.mission_item_send(
            master.target_system,
            master.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 0, 0, 0,
            lat_i, lon_i, alt_i
        )
        print(f"📍 Sent waypoint #{seq}: {lat_i:.6f}, {lon_i:.6f}, alt={alt_i}")

    # Wait for ACK and measure time
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    ack_ts = time.time() if ack else None

    if ack:
        latency_ms = (ack_ts - sent_ts) * 1000.0
        print(f"✅ MISSION_ACK received → E2E latency: {latency_ms:.2f} ms")
        # log to CSV
        if not os.path.exists(LATENCY_LOG):
            with open(LATENCY_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["sent_ts", "ack_ts", "latency_ms", "lat", "lon"])
        with open(LATENCY_LOG, "a", newline="") as f:
            csv.writer(f).writerow([f"{sent_ts:.6f}", f"{ack_ts:.6f}", f"{latency_ms:.3f}", lat, lon])
    else:
        print("❌ No MISSION_ACK received from PX4.")


# ------------------------------
# MQTT message handler
# ------------------------------
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode().strip()
        parts = payload.split(',')

        if len(parts) < 2:
            print(f"⚠️ Invalid MQTT message: {payload}")
            return

        lat = float(parts[0])
        lon = float(parts[1])
        sent_ts = float(parts[2]) if len(parts) >= 3 else time.time()

        save_waypoint_to_file(lat, lon, ALTITUDE)
        upload_mission_to_px4(userdata["px4_master"], sent_ts, lat, lon, ALTITUDE)

    except Exception as e:
        print(f"⚠️ Error processing message: {e}")
        print("Raw:", msg.payload.decode())


# ------------------------------
# Main
# ------------------------------
def main():
    px4_master = connect_px4()

    client = mqtt.Client(userdata={"px4_master": px4_master})
    client.on_message = on_message

    while True:
        try:
            print(f"📡 Connecting to MQTT broker at {MQTT_BROKER} ...")
            client.connect(MQTT_BROKER, 1883, 60)
            break
        except Exception as e:
            print(f"⚠️ MQTT connect failed: {e}")
            time.sleep(2)

    client.subscribe(MQTT_TOPIC)
    print(f"🛰️ Subscribed to topic: {MQTT_TOPIC}")
    print("🛫 Ready to inject waypoints + measure latency")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Exiting ...")
        client.disconnect()


if __name__ == "__main__":
    main()
