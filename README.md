# 🚁 Edge AI-Enabled Autonomous UAV for Real-Time Flood Detection & Geospatial GPS Mapping in Disaster Response

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

<img width="1002" height="451" alt="newfm_final (2)" src="https://github.com/user-attachments/assets/522152d7-048a-4e15-9360-2a3ec040e213" />


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

![ChatGPT Image May 30, 2025, 08_49_15 PM](https://github.com/user-attachments/assets/8efcc8b4-8248-461e-ae17-c64ada0b727f)

---

## 📊 Dataset

Real-world dataset sourced from:

🔗 [https://github.com/sohailahmedkhan/Flood-Detection-from-Images-using-Deep-Learning](https://github.com/sohailahmedkhan/Flood-Detection-from-Images-using-Deep-Learning)

Used for:

* ResNet18 (classification)
* DeepLabv3+ (segmentation)

---


## Flood UAV Simulation Setup

## 1. System Requirements

* Ubuntu 20.04 or 22.04 (compatible Linux distribution)
* ROS 2 Humble
* PX4 Autopilot
* Gazebo Classic
* Python 3.8+
* NVIDIA Jetson Nano (for deployment)

---

## 2. System Setup Instructions

### 2.1 Update Locale

To avoid locale-related issues, run:

```bash
sudo apt update
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

---

### 2.2 Add ROS 2 Package Repository

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
```

---

### 2.3 Install ROS 2 Humble Desktop

```bash
sudo apt update && sudo apt install ros-humble-desktop
```

---

### 2.4 Source ROS 2 Environment

```bash
source /opt/ros/humble/setup.bash
```

To make it permanent:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

### 2.5 Install Gazebo Simulator

```bash
sudo apt update && sudo apt install gazebo
```

---

### 2.6 Install ROS 2 Gazebo Plugins

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
```

---

### 2.7 Verify Installation

```bash
gazebo
```

This should launch the Gazebo simulator window.

---

### 2.8 Clone PX4 Autopilot Repository

Navigate to your workspace and clone PX4:

```bash
cd ~/TEEP/src
git clone --recursive https://github.com/PX4/PX4-Autopilot.git
```

Install PX4 dependencies and build:

```bash
cd PX4-Autopilot
make px4_sitl gazebo
```

Access PX4 Gazebo simulation models and worlds:

```bash
cd ~/TEEP/src/PX4-Autopilot/Tools/simulation/gazebo-classic
./sitl_gazebo-classic
```

You can add or find new models/worlds in the `models` and `worlds` directories.

---

## 4. Setup ROS 2 Workspace

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

---

## 5. Install Python Dependencies

```bash
pip install torch torchvision opencv-python numpy
pip install paho-mqtt
```

---

## 6. PX4 SITL Setup

```bash
cd PX4-Autopilot
make px4_sitl gazebo
```
<img width="1897" height="987" alt="Screenshot 2025-08-05 103054_edited" src="https://github.com/user-attachments/assets/05dc9a1e-2bf9-44af-ac04-a371b711d027" />

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
<img width="786" height="472" alt="imp" src="https://github.com/user-attachments/assets/bfe345d4-150b-4303-a44f-97e4c45c2ad5" />

---

## 🚀 Autonomous UAV Waypoint Injection & Mission Execution

To enable fully autonomous navigation in flood-affected areas, this system implements a **real-time waypoint injection pipeline** that integrates:

- Onboard AI perception  
- Dynamic mission planning  
- PX4 flight control  

Traditional waypoint planning is manual and slow, limiting responsiveness during disaster scenarios. This system enables **real-time waypoint updates with minimal latency**, allowing the UAV to dynamically adapt to newly detected flood regions.

---

### 🔄 System Workflow

<img width="312" height="351" alt="mqtt_to_px4" src="https://github.com/user-attachments/assets/c057bb44-711d-48b4-98b9-eea6e1615fb5" />


1. **Flood Detection (Onboard AI)**
   - Images from the UAV camera are processed on the **Jetson Nano**
   - Deep learning models detect flood regions in real time

2. **GPS Conversion**
   - Detected flood centers are converted into GPS coordinates  
   - Published to ROS 2 topic:
     ```
     /next_gps_waypoint
     ```
   - Message type: `GeoPoseStamped`

3. **Real-Time GeoTask Dispatcher**
   - Implemented in:
     ```
     mqtt_to_px4.py
     ```
   - Subscribes to:
     ```
     uav/flood_detection (MQTT)
     ```
   - Responsibilities:
     - Convert GPS → PX4 waypoints  
     - Generate QGroundControl (QGC) mission files  
     - Update PX4 mission dynamically  

4. **PX4 Mission Update (MAVROS Services)**
   - Clear mission:
     ```
     WaypointClear
     ```
   - Upload mission:
     ```
     WaypointPush
     ```

5. **Autonomous Navigation**
   - UAV navigates toward detected flood regions  
   - PX4 handles:
     - Attitude control  
     - Altitude stabilization  
     - Velocity control  

6. **Latency Monitoring**
   - Timestamps logged across pipeline  
   - Ensures real-time performance  

---

### 🔁 Closed-Loop Autonomy


Camera → AI Detection → GPS Conversion → Waypoint Injection → PX4 Control → UAV Motion


✔ Enables fully autonomous flood detection and response  
✔ Dynamic mission adaptation  
✔ Fault-tolerant operation  

---

## 🧪 Experiments

### A. Simulation (PX4 SITL + Gazebo)

The system was validated in a **PX4 SITL-Gazebo environment**.

<img width="1920" height="1030" alt="21 04 2025_13 08 09_REC" src="https://github.com/user-attachments/assets/bd5df132-b713-447d-9288-60bfde8a5308" />


---

### B. Real-World Deployment

After simulation, the system was deployed on real hardware.
<img width="1536" height="1024" alt="ChatGPT Image May 17, 2025, 01_33_03 PM" src="https://github.com/user-attachments/assets/286e55d7-4a3f-4e22-896f-5d24c643b3d2" />

#### 🧩 Hardware
- Pixhawk Flight Controller  
- NVIDIA Jetson Nano  

#### ⚙️ Deployment
- ROS 2 nodes run on Jetson Nano  
- Same pipeline as simulation  

#### 🔄 Autonomous Operation
- Camera → AI detection → GPS conversion  
- Real-time waypoint updates  
- PX4 mission updated dynamically  

#### 🎯 Mission Flow


TAKEOFF → WAYPOINTS → RTL

<img width="363" height="376" alt="fly" src="https://github.com/user-attachments/assets/39443e08-bed7-48e2-9267-1d3405b90cb1" />
<img width="1206" height="595" alt="20 05 2025_18 51 28_REC" src="https://github.com/user-attachments/assets/a4ad1a12-2098-40cf-bcd4-3729cf3df7df" />


---

## 📌 Key Observations

- Inference is the primary bottleneck in baseline (>95%)  
- TensorRT enables **real-time performance (<20 ms)**  
- DeepLabv3+ preferred for:
- Better real-world segmentation accuracy  

- U-Net preferred for:
- Stable high-frequency simulation  

---

## 🔍 Inference Latency Insights

- Preprocessing (~13 ms) and GPS conversion (~4 ms)  
originally contributed **<2% latency**  

After optimization, further improvements observed due to:

- Reduced Python overhead (async execution)  
- Zero-copy memory reuse  
- Parallel execution across threads  

This module, combined with the **Real-Time GeoTask Dispatcher**, forms the **core intelligence layer** of the UAV flood monitoring system.

---

## 📸 Results

* Real-time flood segmentation
* Autonomous waypoint navigation
* Dynamic mission updates
* Stable UAV flight with AI control
* No human intervention required

---

## 🔬 Research Contributions

* End-to-end **perception-to-action UAV pipeline**
* Real-time **AI + flight control integration**
* **Latency-optimized edge inference**
* Dynamic **GPS waypoint injection**
* Fault-tolerant autonomous UAV system

---

## 👨‍💻 Authors

* Ayush Kumar
* Huang Po Chun
* Ayush Pratap
* Pao-Ann Hsiung

Department of Computer Science and Information Engineering,
National Chung Cheng University, Taiwan

