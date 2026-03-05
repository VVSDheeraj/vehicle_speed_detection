import cv2
import numpy as np
import os

# Function to calculate speed from pixel per second to kilometers per hour
# To calculate speed from pixel per second to km/hr
# speed(km/hr) = speed(pps) * pixel_per_meter * frames_per_second * 3.6
# Assuming distance between two lines (red and blue to be 40 meters)
# The two lines implementation has been done in the YOLO script(speed_estimate.py)
# Number of pixels between the two lines is 70
def calculate_speed(flow, mask):
    distance = 10
    number_of_pixels = 70
    # Calculating the speed based on the magnitude of optical flow vectors within the detected object's mask
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    speed = np.mean(magnitude[mask == 255])
    pixels_per_metre = distance/number_of_pixels
    speed_meter_per_second = pixels_per_metre * speed * fps
    # Convert from meters/second to km/hr
    return speed_meter_per_second * 3.6

# Initializing the Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

cap = cv2.VideoCapture('highway.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
_, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Defining a region of interest (ROI) as (x, y, width, height)
# Region of interest to detect the speeds of vehicles which can be represented by a significant number of pixels in a frame
# As the vehicle moves away from the camera the number of pixels representing the vehicle in a single frame is reduced
## Hence even though the vehicle speed is not decreasing, the speed detected by optical flow will be less
# As optical flow depends on the number of pixels changed in position in every frame to detect motion.
roi_x, roi_y, roi_width, roi_height = 0, 500, 1920, 400 
count = 0
# Creating a folder to save all the frames
if not os.path.exists('detected_frames_opticalflow'):
    os.makedirs('detected_frames_opticalflow')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_optical_flow.avi', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(frame2)

    # Limiting the foreground mask to the ROI
    fgMask[:roi_y, :] = 0
    fgMask[roi_y+roi_height:, :] = 0
    fgMask[:, :roi_x] = 0
    fgMask[:, roi_x+roi_width:] = 0

    # Applying optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # To get cleaner object segmentation in the video frames
    # Applying additional morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)

    # Finding contours on the cleaned mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1750:  # Adjusting the threshold to an appropriate value to detect only a large group of pixels
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            contour_mask = np.zeros_like(cleaned_mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            speed = calculate_speed(flow, contour_mask)
            cv2.putText(frame2, f"{speed:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    prev_gray = gray

    out.write(frame2)

    # Saving the frames
    frame_filename = f'detected_frames_opticalflow/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame2)
    
    count += 1

cap.release()
out.release()
# cv2.destroyAllWindows()  # Commented out - requires GUI support
