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

<img width="746" height="571" alt="14 04 2025_13 47 51_REC" src="https://github.com/user-attachments/assets/bc71e7b6-3835-4c5d-9926-9adf02241f35" />

<img width="1437" height="938" alt="14 04 2025_14 35 07_REC" src="https://github.com/user-attachments/assets/8dad2a30-2623-4d53-b72f-a482e0eddf48" />



### 2. U-Net (Simulation Segmentation)

* IoU: **0.932**
* Dice: **0.956**
* Pixel Accuracy: **99.63%**

<img width="1125" height="722" alt="18 04 2025_18 51 35_REC" src="https://github.com/user-attachments/assets/c1ec2983-3995-4d5b-b68f-84d7d7b55fd6" />

<img width="1304" height="877" alt="19 04 2025_11 23 32_REC" src="https://github.com/user-attachments/assets/d80df5ee-0ed9-444b-b7b3-fcde93b7edeb" />



### 3. DeepLabv3+ (MobileNetV3 Backbone)

* Real UAV imagery
* Pixel Accuracy: **89.26%**
* Optimized for real-world deployment


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


## 🚀 Running the System

### 🔹 Step 1: Launch PX4 + Gazebo

```bash
make px4_sitl gazebo
```
<img width="1897" height="987" alt="Screenshot 2025-08-05 103054_edited" src="https://github.com/user-attachments/assets/05dc9a1e-2bf9-44af-ac04-a371b711d027" />

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
<img width="786" height="472" alt="imp" src="https://github.com/user-attachments/assets/bfe345d4-150b-4303-a44f-97e4c45c2ad5" />

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

Unet:

<img width="600" height="500" alt="confusion_matrix_unet" src="https://github.com/user-attachments/assets/61aeca2b-c668-4c3a-86b5-e7114e8a88dc" />


ResNet18:

<img width="600" height="500" alt="confusion_matrix_resnet18" src="https://github.com/user-attachments/assets/939a0b93-0e34-4c73-9460-48b42e6acd06" />

DeepLabV3+ (Backbone with MobileNet)

<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/ef22963e-d714-4b8c-87e4-6e66543d0a30" />


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

## 👨‍💻 Authors

* Ayush Kumar
* Huang Po Chun
* Ayush Pratap
* Pao-Ann Hsiung

Department of Computer Science and Information Engineering
National Chung Cheng University, Taiwan

