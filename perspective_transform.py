"""
Perspective transform for bird's-eye view speed detection
Calibrated using autobahn delineators: 41m wide × 150m deep
"""
import cv2
import numpy as np

class PerspectiveTransformer:
    
    def __init__(self, src_points, dst_points, output_width=800, output_height=600):
        self.src_points = np.float32(src_points)
        self.dst_points_meters = np.float32(dst_points)
        self.output_width = output_width
        self.output_height = output_height
        
        self.real_width = max(dst_points[1][0], dst_points[3][0])
        self.real_height = max(dst_points[2][1], dst_points[3][1])
        
        self.pixels_per_meter_x = output_width / self.real_width
        self.pixels_per_meter_y = output_height / self.real_height
        self.pixels_per_meter = (self.pixels_per_meter_x + self.pixels_per_meter_y) / 2
        self.meters_per_pixel = 1.0 / self.pixels_per_meter
        
        self.dst_points_pixels = np.float32([
            [dst_points[0][0] * self.pixels_per_meter_x, dst_points[0][1] * self.pixels_per_meter_y],
            [dst_points[1][0] * self.pixels_per_meter_x, dst_points[1][1] * self.pixels_per_meter_y],
            [dst_points[2][0] * self.pixels_per_meter_x, dst_points[2][1] * self.pixels_per_meter_y],
            [dst_points[3][0] * self.pixels_per_meter_x, dst_points[3][1] * self.pixels_per_meter_y]
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points_pixels)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points_pixels, self.src_points)
        
        print(f"Perspective Transform Initialized:")
        print(f"  Real-world area: {self.real_width:.2f}m × {self.real_height:.2f}m")
        print(f"  Output image: {output_width}px × {output_height}px")
        print(f"  Scale: {self.pixels_per_meter:.2f} pixels/meter")
        print(f"  Resolution: {self.meters_per_pixel:.4f} meters/pixel")
    
    def transform_frame(self, frame):
        return cv2.warpPerspective(frame, self.M, (self.output_width, self.output_height))
    
    def transform_point(self, point):
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.M)
        return tuple(transformed[0][0])
    
    def inverse_transform_point(self, point):
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.M_inv)
        return tuple(transformed[0][0])
    
    def transform_points(self, points):
        if len(points) == 0:
            return []
        
        points_array = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points_array, self.M)
        return [tuple(pt) for pt in transformed[0]]
    
    def pixels_to_meters(self, pixel_distance):
        return pixel_distance * self.meters_per_pixel
    
    def meters_to_pixels(self, meter_distance):
        return meter_distance * self.pixels_per_meter
    
    def draw_calibration_overlay(self, frame, color=(0, 255, 0), thickness=2):
        result = frame.copy()
        
        pts = self.src_points.astype(np.int32)
        cv2.polylines(result, [pts], True, color, thickness)
        
        labels = ['TL (0,0)', 'TR (41m,0)', 'BL (0,50m)', 'BR (41m,50m)']
        for i, (pt, label) in enumerate(zip(pts, labels)):
            cv2.circle(result, tuple(pt), 5, color, -1)
            cv2.putText(result, label, (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(result, f"Calibration: {self.real_width:.0f}m x {self.real_height:.0f}m", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"{self.meters_per_pixel:.4f} m/px in bird's-eye view", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def get_grid_overlay(self, frame, grid_spacing_meters=1.0):
        result = frame.copy()
        
        for x_meters in np.arange(0, self.real_width, grid_spacing_meters):
            x_pixels = int(x_meters * self.pixels_per_meter_x)
            cv2.line(result, (x_pixels, 0), (x_pixels, self.output_height),
                    (100, 100, 100), 1)
            cv2.putText(result, f"{x_meters:.0f}m", (x_pixels + 2, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for y_meters in np.arange(0, self.real_height, grid_spacing_meters):
            y_pixels = int(y_meters * self.pixels_per_meter_y)
            cv2.line(result, (0, y_pixels), (self.output_width, y_pixels),
                    (100, 100, 100), 1)
            cv2.putText(result, f"{y_meters:.0f}m", (5, y_pixels - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result


def create_transformer_from_config():
    from config import (PERSPECTIVE_SRC_POINTS, PERSPECTIVE_DST_POINTS,
                        PERSPECTIVE_OUTPUT_WIDTH, PERSPECTIVE_OUTPUT_HEIGHT)
    
    return PerspectiveTransformer(
        PERSPECTIVE_SRC_POINTS,
        PERSPECTIVE_DST_POINTS,
        PERSPECTIVE_OUTPUT_WIDTH,
        PERSPECTIVE_OUTPUT_HEIGHT
    )


if __name__ == "__main__":
    print("Testing Perspective Transform Module\n")
    
    transformer = create_transformer_from_config()
    
    test_point = (636, 236)
    transformed = transformer.transform_point(test_point)
    print(f"\nTest point transformation:")
    print(f"  Original: {test_point}")
    print(f"  Transformed: ({transformed[0]:.1f}, {transformed[1]:.1f}) pixels")
    print(f"  In real-world: (0.0, 0.0) meters")
    
    print(f"\nDistance conversions:")
    print(f"  100 pixels in bird's-eye = {transformer.pixels_to_meters(100):.2f} meters")
    print(f"  1.0 meters = {transformer.meters_to_pixels(1.0):.1f} pixels")
    
    print("\nTransformer ready for use!")
