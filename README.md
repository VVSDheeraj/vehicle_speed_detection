# Vehicle Speed Detection using Optical Flow and YOLO

This project implements vehicle speed detection using two different approaches: **Optical Flow** methods and **YOLO object detection**. It processes highway traffic videos to detect, track, and estimate the speed of vehicles.

## Project Overview

The project uses computer vision techniques to:
- Detect vehicles in video footage
- Track their movement across frames
- Calculate their speed based on pixel displacement and distance assumptions
- Count vehicles moving in different directions

## Files Description

### Main Scripts

1. **lukas_kanade.py**
   - Implements vehicle speed detection using the **Lucas-Kanade optical flow** method
   - Tracks sparse feature points on vehicles to estimate motion
   - Faster but less dense optical flow computation
   - **Output**: `output_lk_optical.mp4` video file

2. **gunnar_farneback.py**
   - Implements vehicle speed detection using the **Gunnar-Farneback optical flow** method
   - Computes dense optical flow across entire frames
   - More accurate for complex scenes but computationally intensive
   - **Outputs**: 
     - `detected_frames_opticalflow/` folder with annotated frames and speed data
     - `output_optical_flow.avi` combined output video

3. **yolo.py**
   - Uses **YOLO 26** (latest version) for object detection and vehicle tracking
   - Draws two reference lines (red and blue) representing a 10-meter distance assumption
   - Counts vehicles crossing the lines in both directions (up and down)
   - Demonstrates distance/speed calculation concept using YOLO detections
   - Depends on `tracker.py` for object tracking
   - **Outputs**:
     - `detected_frames_yolo/` folder with annotated frames showing counters and reference lines
     - `output_yolo.avi` combined output video

4. **tracker.py**
   - Helper module containing the `Tracker` class for object tracking
   - Used by `yolo.py` to track detected vehicles across frames
   - Assigns unique IDs to vehicles and maintains their positions
   - **Cannot be executed standalone** - only imported by other scripts


### Output Directories

- **detected_frames_opticalflow/** - Individual frames from Gunnar-Farneback method
- **detected_frames_yolo/** - Individual frames from YOLO detection
- **__pycache__/** - Python bytecode cache directory

## Prerequisites

### Required Input
- **highway.mp4** - Input video file (must be in the project directory)

### Dependencies
Install required packages:
```bash
pip install ultralytics opencv-python pandas numpy matplotlib scipy
```

## How to Run

### 1. Lucas-Kanade Optical Flow Method
```bash
python lukas_kanade.py
```
**Output**: `output_lk_optical.mp4`

### 2. Gunnar-Farneback Optical Flow Method
```bash
python gunnar_farneback.py
```
**Outputs**: 
- Folder: `detected_frames_opticalflow/`
- Video: `output_optical_flow.avi`

### 3. YOLO Detection Method
```bash
python yolo.py
```
**Outputs**:
- Folder: `detected_frames_yolo/`
- Video: `output_yolo.avi`

## Technical Details

### Distance Assumption
- The scripts assume a **10-meter distance** between two reference lines drawn on the video
- Speed calculation uses: **Speed = Distance / Time**
- Pixel-to-meter conversion is based on this assumption

### YOLO Vehicle Counting
- Red and blue lines serve as virtual gates
- Vehicles crossing these lines are counted separately for upward and downward movement
- Counter is displayed on the video frames

### Model Update
- Project upgraded from YOLOv8 to **YOLO 26** (latest version as of March 2026)
- Improved accuracy and performance compared to previous versions

## Notes

- Ensure all scripts and `highway.mp4` are in the same directory
- `tracker.py` must be present for `yolo.py` to function
- YOLO models will auto-download on first run if not present
- Processing time varies based on video length and chosen method
- Optical flow methods are more CPU-intensive than YOLO detection


