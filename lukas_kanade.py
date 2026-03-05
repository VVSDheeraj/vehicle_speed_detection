import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate speed
def calculate_speed(previous_position, current_position, fps, pixels_per_meter):
    # Computing the Euclidean distance between the previous and current positions
    distance_pixels = np.sqrt((current_position[0] - previous_position[0]) ** 2 + (current_position[1] - previous_position[1]) ** 2)
    distance_meters = distance_pixels / pixels_per_meter  # Convert pixels to meters
    speed_m_per_s = distance_meters * fps  # Calculate speed in meters per second
    speed_kmh = speed_m_per_s * 3.6  # Convert meters/second to km/h
    return speed_kmh

# Opening the video file
cap = cv2.VideoCapture('highway.mp4')

# Reading the first frame and converting it to grayscale
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)
out1 = cv2.VideoWriter('output_lk_optical.mp4', fourcc1, fps, (first_frame.shape[1], first_frame.shape[0]))

# Shi-Tomasi corner detection parameters
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detecting the initial points
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask for drawing purposes
mask = np.zeros_like(first_frame)

pixels_per_meter = 10 / 70

# Frame rate
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

        # Filtering out points with speed less than 1 km/h
        filtered_new = []
        filtered_old = []
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()

            speed = calculate_speed((c, d), (a, b), fps, pixels_per_meter)

            if speed >= 10:  # Filtering out points with speed less than 1 km/h
                filtered_new.append(new)
                filtered_old.append(old)

        filtered_new = np.array(filtered_new)
        filtered_old = np.array(filtered_old)

        # Merging points within 200 pixels
        used_points = set()
        for i in range(len(filtered_new)):
            if i in used_points:
                continue

            a, b = filtered_new[i].ravel()
            c, d = filtered_old[i].ravel()
            cluster_points_new = [filtered_new[i]]
            cluster_points_old = [filtered_old[i]]

            for j in range(i + 1, len(filtered_new)):
                if j in used_points:
                    continue

                a2, b2 = filtered_new[j].ravel()
                if np.sqrt((a - a2) ** 2 + (b - b2) ** 2) < 300:
                    cluster_points_new.append(filtered_new[j])
                    cluster_points_old.append(filtered_old[j])
                    used_points.add(j)

            cluster_center_new = np.mean(cluster_points_new, axis=0)
            cluster_center_old = np.mean(cluster_points_old, axis=0)

            speed = calculate_speed(cluster_center_old, cluster_center_new, fps, pixels_per_meter)

            # Drawing speed and a 100x100 box on the image
            a, b = cluster_center_new.ravel()
            cv2.putText(frame, f'{speed:.2f} km/h', (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            # Drawing a 20x20 box
            top_left = (int(a) - 50, int(b) - 50)
            bottom_right = (int(a) + 50, int(b) + 50)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Updating the previous frame and points
        prev_gray = gray.copy()
        prev_points = filtered_new.reshape(-1, 1, 2)

    # Periodically re-detecting the new feature points
    if frame_idx % 10 == 0 or prev_points is None or len(prev_points) == 0:
        new_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if new_points is not None:
            if prev_points is not None and len(prev_points) > 0:
                prev_points = np.concatenate((prev_points, new_points), axis=0)
            else:
                prev_points = new_points

    frame_idx += 1

    # Displaying the result
    out1.write(frame)
    
cap.release()
out1.release()