"""
Config for vehicle speed detection
Calibrate these values based on your video/camera setup
"""

# Perspective transform - bird's-eye view for better accuracy
ENABLE_PERSPECTIVE_TRANSFORM = True

# Source points in video (1920x1080)
# Based on 50m autobahn delineators - order: TL, TR, BL, BR
PERSPECTIVE_SRC_POINTS = [
    [616, 300],
    [1240, 318],
    [9, 531],
    [1833, 531]
]

# Real-world dimensions: 41m wide × 105m deep
PERSPECTIVE_DST_POINTS = [
    [0, 0],
    [41, 0],
    [0, 105],
    [41, 105]
]

# Output resolution: 30px per meter
PERSPECTIVE_OUTPUT_WIDTH = 1230
PERSPECTIVE_OUTPUT_HEIGHT = 3150

# Fallback calibration when perspective transform is disabled
CALIBRATION_DISTANCE_METERS = 10
CALIBRATION_DISTANCE_PIXELS = 70
REFERENCE_Y_POSITION = 650
REFERENCE_Y_POSITION_LK = 350

# Speed limits for filtering noise
MIN_VALID_SPEED = 1
MAX_VALID_SPEED = 250

# YOLO two-line detection method
YOLO_RED_LINE_Y = 198
YOLO_BLUE_LINE_Y = 268
YOLO_LINE_OFFSET = 6
YOLO_LINE_DISTANCE_METERS = 10

# Gunnar-Farneback optical flow
GF_ROI_X = 0
GF_ROI_Y = 500
GF_ROI_WIDTH = 1920
GF_ROI_HEIGHT = 400
GF_MIN_CONTOUR_AREA = 1750

GF_PYR_SCALE = 0.5
GF_LEVELS = 3
GF_WINSIZE = 15
GF_ITERATIONS = 3
GF_POLY_N = 5
GF_POLY_SIGMA = 1.2

# Lucas-Kanade optical flow
LK_MAX_CORNERS = 100
LK_QUALITY_LEVEL = 0.3
LK_MIN_DISTANCE = 7
LK_WIN_SIZE = 15
LK_MAX_LEVEL = 2
LK_MIN_SPEED_THRESHOLD = 10
LK_CLUSTER_DISTANCE = 300
LK_REDETECT_INTERVAL = 10


# Output settings
OUTPUT_FPS = 20.0
SAVE_FRAMES = True
OUTPUT_DIR_OPTICALFLOW = 'detected_frames_opticalflow'
OUTPUT_DIR_YOLO = 'detected_frames_yolo'

# Perspective correction
ENABLE_PERSPECTIVE_CORRECTION = True
PERSPECTIVE_SCALE_MIN = 0.5
PERSPECTIVE_SCALE_MAX = 2.0

# Display
FONT_SCALE = 0.6
FONT_THICKNESS = 2
SPEED_DECIMAL_PLACES = 1

def get_metres_per_pixel():
    return CALIBRATION_DISTANCE_METERS / CALIBRATION_DISTANCE_PIXELS

def validate_speed(speed_kmh):
    return MIN_VALID_SPEED <= speed_kmh <= MAX_VALID_SPEED

def apply_perspective_scale(base_scale, y_position, reference_y):
    if not ENABLE_PERSPECTIVE_CORRECTION:
        return base_scale
    
    scale_factor = y_position / reference_y if reference_y > 0 else 1.0
    scale_factor = max(PERSPECTIVE_SCALE_MIN, min(scale_factor, PERSPECTIVE_SCALE_MAX))
    return base_scale * scale_factor

"""
Calibration guide:

1. Measure a known distance in your video (lane width, road markings, etc)
2. Count the pixel distance and update CALIBRATION_DISTANCE_* values
3. For YOLO: set line positions and measure real distance between them
4. Test with known speeds and adjust if needed
"""

