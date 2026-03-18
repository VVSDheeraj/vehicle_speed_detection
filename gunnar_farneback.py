import cv2
import numpy as np
import os

# Simple speed calculation from optical flow
def calculate_speed(flow, mask, fps, y_center):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    pixels_per_frame = np.mean(magnitude[mask == 255])
    
    if np.isnan(pixels_per_frame) or pixels_per_frame < 0.1:
        return 0
    
    scale_factor = 400.0 / max(y_center, 200)
    metres_per_pixel = (10.0 / 70.0) * scale_factor
    
    # Calculate speed: (pixels/frame) * (meters/pixel) * (frames/second) = m/s
    speed_m_per_s = pixels_per_frame * metres_per_pixel * fps
    speed_km_per_h = speed_m_per_s * 3.6
    
    if speed_km_per_h < 5 or speed_km_per_h > 200:
        return 0
    
    return speed_km_per_h

# Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

cap = cv2.VideoCapture('highway.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
_, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

count = 0
if not os.path.exists('detected_frames_opticalflow'):
    os.makedirs('detected_frames_opticalflow')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_optical_flow.avi', fourcc, 20.0, 
                      (frame1.shape[1], frame1.shape[0]))

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(frame2)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create mask for this contour
            contour_mask = np.zeros_like(cleaned_mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Calculate speed with simple perspective correction
            y_center = y + h // 2
            speed = calculate_speed(flow, contour_mask, fps, y_center)
            
            # Display if speed is valid
            if speed > 0:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame2, f"{speed:.0f} km/h", 
                          (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    prev_gray = gray

    cv2.putText(frame2, "Gunnar-Farneback Optical Flow",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame2)

    # Saving the frames
    frame_filename = f'detected_frames_opticalflow/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame2)
    
    count += 1

cap.release()
out.release()
print("Processing complete!")
