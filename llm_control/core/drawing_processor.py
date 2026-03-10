import cv2
import numpy as np
import math
import os
from typing import List, Tuple, Dict

class DrawingProcessor:
    def __init__(self):
        self.A4_WIDTH_MM = 170
        self.A4_HEIGHT_MM = 207
        self.MIN_CONTOUR_LENGTH_PX = 30
        self.TIME_ESTIMATE_FACTOR = 0.02
        self.THRESHOLD_OPTIONS = [("Option {}".format(i), i*10, i*20) for i in range(1, 8)]
        
        # Cache for processed commands so start_drawing can retrieve them
        self.cached_options: Dict[str, List[Tuple[float, float, float]]] = {}

    def process_and_cache_image(self, image_path: str) -> str:
        """Processes an image through all thresholds and returns a summary for the LLM."""
        if not os.path.exists(image_path):
            return f"Error: File {image_path} not found."

        self.cached_options.clear()
        summary = f"Image processed successfully. Here are the threshold options:\n"

        for i, (label, t1, t2) in enumerate(self.THRESHOLD_OPTIONS):
            contours_xy, w, h = self._image_to_contours(image_path, t1, t2)
            if not contours_xy or w == 0 or h == 0:
                summary += f"- {label}: No lines detected.\n"
                continue
            
            # Default pen_down_z will be overridden at draw time
            commands = self._create_drawing_paths(contours_xy, w, h, -10.0)
            if commands:
                self.cached_options[label] = commands
                mins = (len(commands) * self.TIME_ESTIMATE_FACTOR) / 60
                summary += f"- {label}: {len(commands)} commands. Est. Time: {mins:.1f} mins.\n"

        if not self.cached_options:
            return "Failed to generate any paths. The image might be too blank or simple."
        
        summary += "\nAsk the user which Option they would like to draw."
        return summary

    def get_commands(self, option_label: str, pen_down_z: float) -> List[Tuple[float, float, float]]:
        """Retrieves cached commands and adjusts the Z-height."""
        if option_label not in self.cached_options:
            return None
        
        original_commands = self.cached_options[option_label]
        adjusted_commands = []
        pen_up_z = pen_down_z / 10 if pen_down_z > 0 else pen_down_z * 1.5

        # Recalculate Z heights based on user's specific pen_down_z preference
        for x, old_z, y in original_commands:
            new_z = pen_down_z if old_z < (pen_up_z - 1) else pen_up_z
            adjusted_commands.append((x, new_z, y))
            
        return adjusted_commands

    def _image_to_contours(self, image_path: str, t1: int, t2: int):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None, 0, 0
        h, w = image.shape[:2]
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, t1, t2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.arcLength(c, closed=False) > self.MIN_CONTOUR_LENGTH_PX]
        
        contours_xy = []
        for contour in filtered:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            contours_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2])
        return contours_xy, w, h

    def _create_drawing_paths(self, contours_xy, image_width, image_height, pen_down_z):
        scale_factor = min(self.A4_WIDTH_MM / image_width, self.A4_HEIGHT_MM / image_height)
        scaled_contours = []
        for contour in contours_xy:
            scaled_contour = [self._scale_point(p, image_width, image_height, scale_factor) for p in contour]
            if len(scaled_contour) >= 1: scaled_contours.append(scaled_contour)
            
        ordered_contours = self._optimize_contour_order(scaled_contours)
        
        pen_up_z = pen_down_z / 10 if pen_down_z > 0 else pen_down_z * 1.5
        robot_commands = []
        
        for contour in ordered_contours:
            if len(contour) == 1:
                pt = contour[0]
                robot_commands.extend([(pt[0], pen_up_z, pt[1]), (pt[0], pen_down_z, pt[1]), (pt[0], pen_up_z, pt[1])])
                continue
            
            start_pt = contour[0]
            robot_commands.extend([(start_pt[0], pen_up_z, start_pt[1]), (start_pt[0], pen_down_z, start_pt[1])])
            for pt in contour[1:]: robot_commands.append((pt[0], pen_down_z, pt[1]))
            robot_commands.append((contour[-1][0], pen_up_z, contour[-1][1]))
            
        return robot_commands

    def _scale_point(self, point_xy, w, h, scale_factor):
        x = (point_xy[0] - (w / 2)) * scale_factor
        y = ((h / 2) - point_xy[1]) * scale_factor
        return (x, y)

    def _optimize_contour_order(self, contours):
        if not contours: return []
        ordered, remaining = [], list(contours)
        current = remaining.pop(0)
        ordered.append(current)
        last_pt = current[-1]

        while remaining:
            best_dist, best_idx, reverse = float('inf'), -1, False
            for i, c in enumerate(remaining):
                d_start = math.dist(last_pt, c[0])
                d_end = math.dist(last_pt, c[-1])
                if d_start < best_dist: best_dist, best_idx, reverse = d_start, i, False
                if d_end < best_dist: best_dist, best_idx, reverse = d_end, i, True
            
            if best_idx != -1:
                next_c = remaining.pop(best_idx)
                if reverse: next_c.reverse()
                ordered.append(next_c)
                last_pt = next_c[-1]
            else: break
        return ordered