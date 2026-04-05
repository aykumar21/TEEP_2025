# 🚁 Edge AI-Enabled Autonomous UAV for Real-Time Flood Detection & GPS Mapping

## 📌 Overview

This project presents an **end-to-end autonomous UAV system** for **real-time flood detection, segmentation, and GPS-based navigation** using **Edge AI**.

The system integrates:

* **PX4 Autopilot (SITL + Hardware)**
* **ROS 2 Humble + MAVROS**
* **Gazebo Simulation**
* **Deep Learning Models (ResNet18, U-Net, DeepLabv3+)**
* **Jetson Nano Edge Deployment**
* **Real-Time GeoTask Dispatcher (MQTT → PX4 Waypoints)**

The UAV can:

* Detect flooded regions in real time
* Segment flood boundaries
* Convert image detections → GPS coordinates
* Dynamically update mission waypoints
* Navigate autonomously without human intervention

---

## 🎯 Key Features

### 🤖 AI-Based Flood Detection

* **ResNet18** → Image-level classification (Flood / Non-Flood)
* **U-Net** → High-precision segmentation (Simulation)
* **DeepLabv3+ (MobileNetV3)** → Real-world UAV segmentation

### ⚡ Real-Time Edge Inference

* TensorRT FP16 optimization
* GPU acceleration (Jetson Nano)
* Multi-threaded ROS 2 pipeline
* Zero-copy memory optimization

### 🛰️ Autonomous Navigation

* Dynamic waypoint injection via MQTT
* PX4 mission update using MAVROS
* Real-time perception → action loop

### 🛡️ Fault-Tolerant Control

* MAVLink heartbeat monitoring
* Offboard mode failure detection
* Automatic fallback:

  * LAND
  * RTL
  * Hover

---

## 🏗️ System Architecture

```
UAV Camera → ROS2 Image Topic
        ↓
Deep Learning Model (Jetson Nano)
        ↓
Flood Detection / Segmentation
        ↓
Pixel → GPS Conversion
        ↓
MQTT (GeoTask Dispatcher)
        ↓
PX4 Waypoint Injection (MAVROS)
        ↓
Autonomous UAV Navigation
```


## 🧠 AI Models

### 1. ResNet18 (Classification)

* Accuracy: **96%**
* Dataset: Hybrid (Real + Gazebo)
* Output: Flood / Non-Flood

### 2. U-Net (Simulation Segmentation)

* IoU: **0.932**
* Dice: **0.956**
* Pixel Accuracy: **99.63%**

### 3. DeepLabv3+ (MobileNetV3 Backbone)

* Real UAV imagery
* Pixel Accuracy: **89.26%**
* Optimized for real-world deployment

---

## 📊 Dataset

Real-world dataset sourced from:

🔗 [https://github.com/sohailahmedkhan/Flood-Detection-from-Images-using-Deep-Learning](https://github.com/sohailahmedkhan/Flood-Detection-from-Images-using-Deep-Learning)

Used for:

* ResNet18 (classification)
* DeepLabv3+ (segmentation)

---

## ⚙️ Installation

### 1. System Requirements

* Ubuntu 20.04
* ROS 2 Humble
* PX4 Autopilot
* Gazebo Classic
* Python 3.8+
* NVIDIA Jetson Nano (for deployment)

---

### 2. Clone Repository

```bash
git clone https://github.com/your-username/flood-uav.git
cd flood-uav
```

---

### 3. Setup ROS 2 Workspace

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

---

### 4. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy
pip install paho-mqtt
```

---

### 5. PX4 SITL Setup

```bash
cd PX4-Autopilot
make px4_sitl gazebo
```

---

## 🚀 Running the System

### 🔹 Step 1: Launch PX4 + Gazebo

```bash
make px4_sitl gazebo
```

---

### 🔹 Step 2: Run MAVROS

```bash
ros2 launch mavros px4.launch.py
```

---

### 🔹 Step 3: Run AI Inference Node

```bash
ros2 run flood_detection deeplab_inference_node
```

---

### 🔹 Step 4: Run GeoTask Dispatcher

```bash
python3 mqtt_to_px4.py
```

---

### 🔹 Step 5: Arm and Start Mission

```bash
ros2 service call /mavros/cmd/arming ...
ros2 service call /mavros/set_mode ...
```

---

## 🧭 Pixel-to-GPS Conversion Pipeline

1. Capture image from UAV camera
2. Run segmentation → binary mask
3. Divide into **4×4 grid**
4. Find highest flood density cell
5. Compute centroid
6. Project to 3D using camera model
7. Convert to GPS (GeographicLib)
8. Publish waypoint

---

## ⚡ Performance & Latency

### 🧠 Inference Performance

| Model      | Before  | After (TensorRT) | Improvement |
| ---------- | ------- | ---------------- | ----------- |
| DeepLabv3+ | 1353 ms | 15.6 ms          | **98.68%**  |
| U-Net      | 355 ms  | 15.4 ms          | **95.82%**  |

---

### ⏱️ End-to-End Latency

* Total pipeline: **0.09 – 0.40 s**
* Real-time performance: **40–50 FPS**

---

## 🛡️ Fault-Tolerant System

Monitors:

* MAVLink heartbeat
* Offboard mode status
* MAVROS health

Fallback Actions:

* LAND
* RTL
* Hover stabilization

---

## 🧪 Simulation vs Real World

| Aspect      | Simulation | Real Deployment |
| ----------- | ---------- | --------------- |
| Model       | U-Net      | DeepLabv3+      |
| Accuracy    | ~99%       | ~89%            |
| Environment | Controlled | Noisy, dynamic  |
| Use Case    | Testing    | Real missions   |

---

## 📸 Results

* Real-time flood segmentation
* Autonomous waypoint navigation
* Dynamic mission updates
* Stable UAV flight with AI control

---

## 🔬 Research Contributions

* End-to-end **perception-to-action UAV pipeline**
* Real-time **AI + flight control integration**
* **Latency-optimized edge inference**
* Dynamic **GPS waypoint injection**
* Fault-tolerant autonomous UAV system

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{ayush2026uavflood,
  title={Edge AI-Enabled Autonomous UAVs for Real-Time Flood Detection and Geospatial GPS Mapping},
  author={Kumar, Ayush and Huang, Po Chun and Pratap, Ayush and Hsiung, Pao-Ann},
  journal={IEEE},
  year={2026}
}
```

---

## 👨‍💻 Authors

* Ayush Kumar
* Huang Po Chun
* Ayush Pratap
* Pao-Ann Hsiung

Department of Computer Science and Information Engineering
National Chung Cheng University, Taiwan

