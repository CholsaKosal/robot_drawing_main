import cv2
import numpy as np
import math
import logging
from typing import List, Tuple

class ClassicImageProcessor:
    """
    Handles the classic Canny edge detection and drawing path generation
    for the robot drawing system.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        self.a4_width_mm = a4_width_mm
        self.a4_height_mm = a4_height_mm
        self.min_contour_length_px = min_contour_length_px

    def image_to_contours(self, image_path_or_array, threshold1, threshold2, save_edge_path=None):
        """
        Convert image to contours using Canny edge detection.
        Accepts a file path or a pre-loaded cv2 image array.
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_path_or_array, np.ndarray):
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY) if len(image_path_or_array.shape) == 3 else image_path_or_array
        else:
            logging.error("Invalid input type for image_to_contours")
            return None, 0, 0

        if image is None:
            logging.error("Could not read or process image input.")
            return None, 0, 0

        image_height, image_width = image.shape[:2]
        if image_height == 0 or image_width == 0:
            logging.error("Invalid image dimensions.")
            return None, 0, 0

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)

        if save_edge_path:
            try:
                cv2.imwrite(save_edge_path, edges)
            except Exception as e:
                logging.error(f"Failed to save edge image to {save_edge_path}: {e}")

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.arcLength(c, closed=False) > self.min_contour_length_px]

        contours_xy = []
        for contour in filtered_contours:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            contours_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2])

        return [c for c in contours_xy if c], image_width, image_height

    def create_drawing_paths(self, contours_xy, image_width, image_height, pen_down_z: float, optimize_paths=True):
        """
        Takes a list of contours (pixel coordinates), scales them to the drawing area,
        optimizes the drawing order, and generates the final robot commands.
        """
        # Calculate a safe pen-up position (higher, i.e., less negative)
        pen_up_z = pen_down_z / 10 if pen_down_z > 0 else pen_down_z * 2.0
        
        if not contours_xy or image_width <= 0 or image_height <= 0:
            return []

        # Calculate scale factor to fit the image to the A4 drawing area
        scale_x = self.a4_width_mm / image_width
        scale_y = self.a4_height_mm / image_height
        scale_factor = min(scale_x, scale_y)

        # Scale all contour points from pixel coordinates to robot-friendly mm coordinates
        scaled_contours = []
        for contour in contours_xy:
            if not contour: continue
            scaled_contour = [self.scale_point_to_a4(p, image_width, image_height, scale_factor) for p in contour]
            if len(scaled_contour) >= 1:
                scaled_contours.append(scaled_contour)

        if not scaled_contours:
            return []

        # Optimize the drawing path to minimize travel distance
        if optimize_paths:
            ordered_contours = self.optimize_contour_order(scaled_contours)
        else:
            ordered_contours = scaled_contours

        # Generate the final list of robot commands (X, Z, Y)
        robot_commands = []
        for contour in ordered_contours:
            if not contour: continue
            
            # Handle single-point contours (dots)
            if len(contour) == 1:
                point = contour[0]
                robot_commands.append((point[0], pen_up_z, point[1]))   # Move to location
                robot_commands.append((point[0], pen_down_z, point[1])) # Pen down
                robot_commands.append((point[0], pen_up_z, point[1]))   # Pen up
                continue

            # Handle multi-point contours (lines)
            start_point = contour[0]
            robot_commands.append((start_point[0], pen_up_z, start_point[1]))   # Move to start of line
            robot_commands.append((start_point[0], pen_down_z, start_point[1])) # Pen down

            # Draw along the contour
            for point in contour[1:]:
                robot_commands.append((point[0], pen_down_z, point[1]))

            # Lift pen at the end of the contour
            final_point = contour[-1]
            robot_commands.append((final_point[0], pen_up_z, final_point[1]))

        return robot_commands
        
    def optimize_contour_order(self, contours: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """
        Sorts contours to minimize travel distance between them using a nearest-neighbor approach.
        """
        if not contours:
            return []

        ordered_contours = []
        remaining_contours = list(contours)
        
        # Start with the first contour
        current_contour = remaining_contours.pop(0)
        ordered_contours.append(current_contour)
        last_point = current_contour[-1]

        while remaining_contours:
            best_dist = float('inf')
            best_idx = -1
            best_reversed = False

            # Find the closest next contour (or the reversed version of it)
            for i, contour in enumerate(remaining_contours):
                dist_start = self.calculate_distance(last_point, contour[0])
                dist_end = self.calculate_distance(last_point, contour[-1])

                if dist_start < best_dist:
                    best_dist, best_idx, best_reversed = dist_start, i, False
                if dist_end < best_dist:
                    best_dist, best_idx, best_reversed = dist_end, i, True

            if best_idx != -1:
                next_contour = remaining_contours.pop(best_idx)
                if best_reversed:
                    next_contour.reverse()
                ordered_contours.append(next_contour)
                last_point = next_contour[-1]
            else:
                logging.warning("Path optimization loop finished unexpectedly.")
                break # Safety break

        return ordered_contours

    def scale_point_to_a4(self, point_xy, image_width, image_height, scale_factor):
        """ Scales and transforms a single (x, y) pixel coordinate to a centered robot coordinate (mm)."""
        x_pixel, y_pixel = point_xy
        x_centered_pixel = x_pixel - (image_width / 2)
        y_centered_pixel = (image_height / 2) - y_pixel  # Invert y-axis for standard Cartesian coordinates
        x_mm = x_centered_pixel * scale_factor
        y_mm = y_centered_pixel * scale_factor
        return (x_mm, y_mm)

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculates Euclidean distance between two points (x, y)."""
        if p1 is None or p2 is None: return float('inf')
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)