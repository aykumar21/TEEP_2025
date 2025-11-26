import paho.mqtt.client as mqtt
import time
import csv
import os

# CSV log file path
LOG_FILE = "latency_log.csv"

# Create file with header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Latency_ms", "Sent_Timestamp", "Received_Timestamp", "Latitude", "Longitude"])

def on_message(client, userdata, msg):
    try:
        # Decode MQTT payload
        payload = msg.payload.decode().strip()
        lat, lon, sent_time = payload.split(',')
        sent_time = float(sent_time)

        # Record receive time
        recv_time = time.time()

        # Calculate latency in milliseconds
        latency_ms = (recv_time - sent_time) * 1000

        # Print latency
        print(f"Latency: {latency_ms:.2f} ms | Message: ({lat}, {lon})")

        # Append to CSV file
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{latency_ms:.2f}", sent_time, recv_time, lat, lon])

    except Exception as e:
        print("Error processing message:", e)
        print("Raw payload:", msg.payload.decode())

# Initialize MQTT client
client = mqtt.Client()
client.connect("192.168.1.93", 1883, 60)  # Replace with Jetson's IP if different
client.subscribe("uav/flood_detection")
client.on_message = on_message

print("📡 Listening for messages... Press Ctrl+C to stop.")
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
