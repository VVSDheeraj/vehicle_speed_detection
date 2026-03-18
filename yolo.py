
"""
YOLO-based vehicle speed detection using two-line method
Tracks vehicles across 105m calibrated distance using perspective transform
Includes Kalman filtering for smooth speed estimation
"""

import cv2
import os
import numpy as np
import torch
import time
import csv
from datetime import datetime
from ultralytics import YOLO
from perspective_transform import create_transformer_from_config


class VehicleKalmanTracker:
    
    # Kalman Filter for tracking vehicle position and velocity in bird's-eye view
    def __init__(self, initial_x, initial_y, fps):
        self.kf = cv2.KalmanFilter(4, 2)
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Measurement matrix: we observe [x, y]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        
        # Transition matrix: constant velocity model
        self.kf.transitionMatrix = np.array([[1, 0, self.dt, 0],
                                              [0, 1, 0, self.dt],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        
        # Process noise - moderate values for highway traffic
        q_pos = 0.05
        q_vel = 1.0
        self.kf.processNoiseCov = np.array([[q_pos, 0, 0, 0],
                                             [0, q_pos, 0, 0],
                                             [0, 0, q_vel, 0],
                                             [0, 0, 0, q_vel]], np.float32)
        
        # Measurement noise
        self.r_xy = 1.0
        self.kf.measurementNoiseCov = np.array([[self.r_xy, 0],
                                                 [0, self.r_xy]], np.float32)
        
        # Initialize state
        self.kf.statePre = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
        self.kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
        
        # Outlier rejection threshold
        self.mahalanobis_threshold = 6.0
        
        self.missed_count = 0
        self.max_missed = 10
        
    def predict(self):
        """Predict next state"""
        return self.kf.predict()
    
    def update(self, measurement_x, measurement_y):
        """Update with measurement and outlier rejection"""
        measurement = np.array([[measurement_x], [measurement_y]], np.float32)
        
        # Mahalanobis distance for outlier rejection
        predicted_measurement = self.kf.measurementMatrix @ self.kf.statePre
        innovation = measurement - predicted_measurement
        
        H = self.kf.measurementMatrix
        P = self.kf.errorCovPre
        R = self.kf.measurementNoiseCov
        S = H @ P @ H.T + R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = np.sqrt(float((innovation.T @ S_inv @ innovation)[0, 0]))
        except:
            mahalanobis_dist = 0
        
        if mahalanobis_dist > self.mahalanobis_threshold:
            self.missed_count += 1
        else:
            self.kf.correct(measurement)
            self.missed_count = 0
        
        state = self.kf.statePost
        return float(state[0][0]), float(state[1][0]), float(state[2][0]), float(state[3][0])
    
    def get_speed_kmh(self):
        """Get speed in km/h from Kalman velocity"""
        vx = float(self.kf.statePost[2][0])
        vy = float(self.kf.statePost[3][0])
        speed_ms = np.sqrt(vx**2 + vy**2)
        return speed_ms * 3.6
    
    def is_lost(self):
        return self.missed_count > self.max_missed

# Check GPU availability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    use_fp16 = True
else:
    use_fp16 = False

model = YOLO('yolo26m-seg.pt')
model.to(device)

cap = cv2.VideoCapture('highway.mp4')

ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit()

# Reset video to beginning
cap.release()
cap = cv2.VideoCapture('highway.mp4')

frame_width, frame_height = 1920, 1080

# Initialize perspective transformer
transformer = create_transformer_from_config()

print("="*50)
print("YOLO + Kalman Filter Speed Detection")
print(f"Resolution: {frame_width}×{frame_height}")
print(f"Scaling: {transformer.meters_per_pixel:.4f} m/pixel")
print("="*50)

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

count = 0
down = {}
up = {}
counter_down = []
counter_up = []
vehicle_speeds = {}
vehicle_kalman_filters = {}
vehicle_ema_speeds = {}
vehicle_logs = {}
vehicle_frame_count = {}
debug_logged_vehicles = set()
EMA_ALPHA = 0.35
MIN_FRAMES_FOR_SPEED = 8

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

print(f"Video FPS: {fps}")
print(f"This means 1 frame = {1/fps:.4f} seconds")
print(f"Speed calculation will use this FPS value directly (no slowdown)\n")

line1_y_meters = 0.0    # First detection line at 0m from top
line2_y_meters = 105.0  # Second line at 105m (bottom edge of calibration)
line_distance_meters = line2_y_meters - line1_y_meters  # 105 meters between lines

# Creating a folder to save frames
if not os.path.exists('detected_frames_yolo'):
    os.makedirs('detected_frames_yolo')

# Create CSV log file
log_filename = f'vehicle_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
log_file = open(log_filename, 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Vehicle_ID', 'Type', 'First_Frame', 'Last_Frame', 'Duration_Sec', 
                     'Avg_Speed_kmh', 'Min_Speed_kmh', 'Max_Speed_kmh', 
                     'Direction', 'Crossed_Lines', 'Total_Detections'])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_yolo.avi', fourcc, fps, (1920, 1080))

if not out.isOpened():
    print("ERROR: Could not create output video file!", flush=True)
    exit(1)

print("Starting video processing...", flush=True)
start_time = time.time()
processing_times = []

while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame", flush=True)
        break
    count += 1
    frame_count += 1
    
    if frame_count == 1:
        print(f"Processing frame 1, shape: {frame.shape}", flush=True)
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        active_vehicles = len([v for v in vehicle_logs.values() if v['last_frame'] >= frame_count - 10])
        print(f"Frame {frame_count} | {avg_fps:.2f} FPS | Active: {active_vehicles} vehicles | "
              f"Down: {len(counter_down)} Up: {len(counter_up)}", flush=True)
    
    try:
        results = model.track(frame, persist=True, verbose=False, device=device, half=use_fp16, conf=0.25)
    except:
        results = model.predict(frame, verbose=False, device=device, half=use_fp16, conf=0.25)
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        cv2.rectangle(frame, (0, 0), (280, 80), (50, 50, 50), -1)
        cv2.putText(frame, f'Down: {len(counter_down)} | Up: {len(counter_up)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        out.write(frame)
        continue
        
    boxes = results[0].boxes.data.cpu().numpy()
    
    track_ids = results[0].boxes.id
    if track_ids is None:
        print(f"Frame {frame_count}: No track IDs, using sequential numbering")
        track_ids = np.arange(len(boxes))
    else:
        track_ids = track_ids.cpu().numpy().astype(int)
    
    for i, box in enumerate(boxes):
        if i >= len(track_ids):
            continue
            
        x3, y3, x4, y4 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        id = int(track_ids[i])
        
        # Filter vehicle classes
        class_id = int(box[6] if box.shape[0] == 7 else box[5])
        class_name = class_list[class_id] if class_id < len(class_list) else "unknown"
        if not any(v in class_name for v in ['car', 'truck', 'bus', 'motorcycle']):
            continue
        
        # Use bbox bottom-center as footpoint
        cx = int((x3 + x4) / 2)
        cy = int(y4)
        
        # Transform to bird's-eye view
        cx_transformed, cy_transformed = transformer.transform_point((cx, cy))
        cx_meters = transformer.pixels_to_meters(cx_transformed)
        cy_meters = transformer.pixels_to_meters(cy_transformed)
        
        # Filter by calibrated area (0 to 105 meters)
        if not (0 <= cy_meters <= 105):
            continue
        
        # Initialize Kalman tracker for new vehicles
        if id not in vehicle_kalman_filters:
            vehicle_kalman_filters[id] = VehicleKalmanTracker(cx_meters, cy_meters, fps)
            vehicle_frame_count[id] = 0
            # Initialize vehicle log
            vehicle_logs[id] = {
                'type': class_name,
                'first_frame': frame_count,
                'last_frame': frame_count,
                'speeds': [],
                'detections': 0,
                'direction': None
            }
        
        # Increment frame count for this vehicle
        vehicle_frame_count[id] += 1
        
        # Get tracker
        tracker = vehicle_kalman_filters[id]
        tracker.predict()
        filtered_x, filtered_y, filtered_vx, filtered_vy = tracker.update(cx_meters, cy_meters)
        
        # Get speed from Kalman filter
        speed_kmh = tracker.get_speed_kmh()
        
        if id not in debug_logged_vehicles and len(debug_logged_vehicles) < 5 and vehicle_frame_count[id] == MIN_FRAMES_FOR_SPEED:
            debug_logged_vehicles.add(id)
            print(f"\n[DEBUG] Vehicle {id} ({class_name}):")
            print(f"  Kalman velocity: vx={filtered_vx:.2f} m/s, vy={filtered_vy:.2f} m/s")
            print(f"  Speed magnitude: {speed_kmh:.1f} km/h")
            print(f"  Position: ({filtered_x:.1f}, {filtered_y:.1f}) meters in bird's-eye view")
            print(f"  FPS used: {fps}")
        
        # Update vehicle log
        vehicle_logs[id]['last_frame'] = frame_count
        vehicle_logs[id]['detections'] += 1
        
        if vehicle_frame_count[id] >= MIN_FRAMES_FOR_SPEED:
            if 30 <= speed_kmh <= 160:
                if id not in vehicle_ema_speeds:
                    vehicle_ema_speeds[id] = speed_kmh
                else:
                    vehicle_ema_speeds[id] = EMA_ALPHA * speed_kmh + (1 - EMA_ALPHA) * vehicle_ema_speeds[id]
                
                vehicle_speeds[id] = vehicle_ema_speeds[id]
                vehicle_logs[id]['speeds'].append(vehicle_speeds[id])
        
        if id not in down and id not in up:
            down[id] = {'min_y': cy_meters, 'max_y': cy_meters, 'avg_vy': filtered_vy, 'frames': 1}
            up[id] = {'min_y': cy_meters, 'max_y': cy_meters, 'avg_vy': filtered_vy, 'frames': 1}
        else:
            if id in down:
                down[id]['min_y'] = min(down[id]['min_y'], cy_meters)
                down[id]['max_y'] = max(down[id]['max_y'], cy_meters)
                down[id]['frames'] += 1
                down[id]['avg_vy'] = 0.6 * down[id]['avg_vy'] + 0.4 * filtered_vy
            if id in up:
                up[id]['min_y'] = min(up[id]['min_y'], cy_meters)
                up[id]['max_y'] = max(up[id]['max_y'], cy_meters)
                up[id]['frames'] += 1
                up[id]['avg_vy'] = 0.6 * up[id]['avg_vy'] + 0.4 * filtered_vy
        
        # Count vehicles that traveled significant distance in one direction
        if id in down:
            travel_range = down[id]['max_y'] - down[id]['min_y']
            avg_vy = down[id]['avg_vy']
            frames_tracked = down[id]['frames']
            # Count as down if: tracked 10+ frames, traveled 35+m, AND clear positive velocity
            if frames_tracked >= 10 and travel_range > 35 and avg_vy > 0.5 and id not in counter_down:
                counter_down.append(id)
                vehicle_logs[id]['direction'] = 'down'
                vehicle_logs[id]['crossed_lines'] = True
        
        if id in up:
            travel_range = up[id]['max_y'] - up[id]['min_y']
            avg_vy = up[id]['avg_vy']
            frames_tracked = up[id]['frames']
            # Count as up if: tracked 10+ frames, traveled 35+m, AND clear negative velocity
            if frames_tracked >= 10 and travel_range > 35 and avg_vy < -0.5 and id not in counter_up:
                counter_up.append(id)
                vehicle_logs[id]['direction'] = 'up'
                vehicle_logs[id]['crossed_lines'] = True
        
        # Draw vehicle
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        
        # Displaying speed and type of the vehicle
        if id in vehicle_speeds:
            speed_text = f"{vehicle_speeds[id]:.0f} km/h"
            type_text = class_name.capitalize()
            cv2.putText(frame, speed_text,
                       (x3, y3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, type_text,
                       (x3, y4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(frame, (0, 0), (280, 80), (50, 50, 50), -1)
    cv2.putText(frame, f'Down: {len(counter_down)} | Up: {len(counter_up)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Frame: {frame_count}', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw detection lines
    overlay = frame.copy()
    line1_points = []
    line2_points = []
    for x_px in range(0, transformer.output_width, 20):
        y1_px = transformer.meters_to_pixels(line1_y_meters)
        y2_px = transformer.meters_to_pixels(line2_y_meters)
        
        pt1 = transformer.inverse_transform_point((x_px, y1_px))
        pt2 = transformer.inverse_transform_point((x_px, y2_px))
        
        if 0 <= pt1[0] < 1920 and 0 <= pt1[1] < 1080:
            line1_points.append(pt1)
        if 0 <= pt2[0] < 1920 and 0 <= pt2[1] < 1080:
            line2_points.append(pt2)
    
    if len(line1_points) > 1:
        pts = np.array(line1_points, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (0, 255, 0), 3)
    
    if len(line2_points) > 1:
        pts = np.array(line2_points, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
    
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Saving the frames
    frame_filename = f'detected_frames_yolo/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)


cap.release()
out.release()

print("\nWriting vehicle logs...")
valid_vehicles = 0
for vid, log in vehicle_logs.items():
    duration = (log['last_frame'] - log['first_frame']) / fps
    crossed = log.get('crossed_lines', False)
    has_speeds = len(log['speeds']) > 0
    
    if duration < 1.0 and not crossed:
        continue
    
    valid_vehicles += 1
    
    if has_speeds:
        avg_speed = np.mean(log['speeds'])
        min_speed = np.min(log['speeds'])
        max_speed = np.max(log['speeds'])
    else:
        avg_speed = min_speed = max_speed = 0
    
    direction = log.get('direction', 'unknown')
    
    csv_writer.writerow([
        vid,
        log['type'],
        log['first_frame'],
        log['last_frame'],
        f"{duration:.2f}",
        f"{avg_speed:.1f}",
        f"{min_speed:.1f}",
        f"{max_speed:.1f}",
        direction,
        'Yes' if crossed else 'No',
        log['detections']
    ])

log_file.close()
print(f"Vehicle log saved to: {log_filename}")
print(f"Total detections: {len(vehicle_logs)}")
print(f"Valid vehicles (>1sec or crossed): {valid_vehicles}")
print(f"Vehicles with speed data: {len([v for v in vehicle_logs.values() if len(v['speeds']) > 0])}")

# Performance statistics
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print("\n" + "="*50)
print("PROCESSING COMPLETE!")
print("="*50)
print(f"Total frames: {frame_count}")
print(f"Processing time: {total_time:.1f}s")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Vehicles down: {len(counter_down)}")
print(f"Vehicles up: {len(counter_up)}")
print(f"Valid vehicles logged (filtered): {valid_vehicles}")
print(f"Output video: output_yolo.avi")
print(f"Vehicle log: {log_filename}")
print("="*50)
