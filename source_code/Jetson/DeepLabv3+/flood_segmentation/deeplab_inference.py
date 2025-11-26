import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models.segmentation as models
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
import math
import time


class DeepLabInference:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use external camera (default is 0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Load DeepLabv3 model
        self.model = models.deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
        self.model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Image transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Camera intrinsics (same as UNet code, adjust if needed)
        self.fx = 277.19
        self.fy = 277.19
        self.cx = 160.5
        self.cy = 120.5

        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(K)

        # GeographicLib object for geodesic calculations
        self.geod = Geodesic.WGS84

        # Rotation matrix for 45° pitch about Y-axis (camera → world)
        theta = np.deg2rad(45)
        self.R_pitch = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Dummy GPS and altitude (for simulation)
        self.current_gps = (37.7749, -122.4194)  # San Francisco
        self.drone_altitude = 3.0  # meters

    def process_frame(self, frame):
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor)['out']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize mask back to original frame size
        mask_resized = cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0])) * 255
        mask_resized = mask_resized.astype(np.uint8)

        # Overlay mask on original image (red channel)
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 2] = mask_resized
        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

        self.draw_grid_and_compute_gps(blended, mask_resized // 255, self.current_gps)

        # Show the result
        cv2.imshow("DeepLabV3 Flood Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def draw_grid_and_compute_gps(self, image, mask, current_gps, grid_size=4):
        height, width = mask.shape
        cell_h, cell_w = height // grid_size, width // grid_size

        max_ratio = -1
        max_cell_center = (0, 0)

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_mask = mask[y1:y2, x1:x2]
                flood_ratio = np.mean(cell_mask)

                # Color code based on flood ratio
                if flood_ratio > 0.5:
                    color = (0, 0, 255)  # Red (High flood)
                elif flood_ratio > 0.2:
                    color = (0, 255, 255)  # Yellow (Moderate flood)
                else:
                    color = (0, 255, 0)  # Green (Safe)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{flood_ratio:.2f}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if flood_ratio > max_ratio:
                    max_ratio = flood_ratio
                    max_cell_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Compute GPS for the most flooded cell
        if max_ratio > 0.3:
            center_x, center_y = max_cell_center
            pixel = np.array([center_x, center_y, 1])

            # Ray from camera through pixel (camera coordinates)
            ray_cam = self.K_inv @ pixel
            ray_cam /= np.linalg.norm(ray_cam)

            # Rotate to world frame
            ray_world = self.R_pitch @ ray_cam
            ray_world /= ray_world[2]

            # Project ray to ground plane (z = 0), given drone altitude h
            h = self.drone_altitude
            scale = -h / ray_world[2]
            ground_point = scale * ray_world

            # Offsets (East, North)
            dx = ground_point[0]
            dy = -ground_point[2]

            # Convert local offset to global GPS using Geodesic.Direct
            lat, lon = current_gps
            distance = np.sqrt(dx**2 + dy**2)
            azimuth = np.rad2deg(np.arctan2(dx, dy))  # bearing from North clockwise
            new_point = self.geod.Direct(lat, lon, azimuth, distance)

            target_lat = new_point['lat2']
            target_lon = new_point['lon2']

            print(f"Flood Detected -> Target GPS: ({target_lat:.6f}, {target_lon:.6f})")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame)

        self.cap.release()


if __name__ == '__main__':
    inference = DeepLabInference()
    inference.run()

