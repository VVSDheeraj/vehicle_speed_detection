import cv2
import numpy as np

# Simple speed calculation with basic perspective correction
def calculate_speed(previous_position, current_position, fps, y_position):
    # Computing the Euclidean distance between positions (pixels per frame)
    distance_pixels = np.sqrt((current_position[0] - previous_position[0]) ** 2 + 
                             (current_position[1] - previous_position[1]) ** 2)
    
    # Simple perspective correction: closer objects (higher y) appear larger
    # Calibration: 10 meters = 70 pixels at y=400
    scale_factor = 400.0 / max(y_position, 200)
    metres_per_pixel = (10.0 / 70.0) * scale_factor
    
    distance_meters = distance_pixels * metres_per_pixel
    speed_m_per_s = distance_meters * fps  # meters per second
    speed_kmh = speed_m_per_s * 3.6  # km/h
    
    # Validate speed
    if speed_kmh < 5 or speed_kmh > 200:
        return 0
    
    return speed_kmh

cap = cv2.VideoCapture('highway.mp4')

ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out1 = cv2.VideoWriter('output_lk_optical.mp4', fourcc1, fps, 
                       (first_frame.shape[1], first_frame.shape[0]))

# Shi-Tomasi corner detection parameters
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detecting the initial points
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_points is not None and len(prev_points) > 0:
        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        # Selecting the good points
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        # Draw motion vectors and speed for each point
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()

            # Calculate speed with simple perspective correction
            speed = calculate_speed((c, d), (a, b), fps, b)

            if speed >= 10:  # Only show vehicles moving at reasonable speed
                cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.putText(frame, f'{speed:.0f} km/h', 
                          (int(a), int(b) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_gray = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)

    # Periodically re-detect new feature points
    if frame_idx % 10 == 0 or prev_points is None or len(prev_points) < 10:
        new_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if new_points is not None:
            if prev_points is not None and len(prev_points) > 0:
                prev_points = np.concatenate((prev_points, new_points), axis=0)
            else:
                prev_points = new_points

    frame_idx += 1

    cv2.putText(frame, "Lucas-Kanade Optical Flow",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out1.write(frame)
    
cap.release()
out1.release()
print("Processing complete!")