import cv2
import numpy as np
import logging
from classic_image_processor import ClassicImageProcessor

class FastEyeTierProcessor(ClassicImageProcessor):
    """
    Mode 5: Fast Topographical + Interactive Smart Fill.
    Uses PolyDP approximation to speed up robot drawing time, allows adjustable
    tiers and detail passes, and fills dark regions ONLY if the user clicks on them.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, max(10, min_contour_length_px // 2))

    def image_to_fast_eye_tiers(self, image_path_or_array, num_tiers, filter_mode=2, save_edge_path=None, user_eye_points=None):
        """
        Slices image into adjustable tiers, approximates paths for speed, 
        and fills dark regions based on interactive user coordinate clicks.
        """
        if user_eye_points is None:
            user_eye_points = []

        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_path_or_array, np.ndarray):
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY) if len(image_path_or_array.shape) == 3 else image_path_or_array
        else:
            logging.error("Invalid input type")
            return None, 0, 0

        if image is None:
            return None, 0, 0

        image_height, image_width = image.shape[:2]
        all_paths_xy = []
        preview = np.full((image_height, image_width), 255, dtype=np.uint8)

        # 1. OPTIMIZE FOR SPEED & TOPOGRAPHY
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(image)
        
        # Apply the chosen Detail Edge Pass
        if filter_mode == 1:
            smoothed = cv2.bilateralFilter(enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
        elif filter_mode == 2:
            smoothed = cv2.bilateralFilter(enhanced_image, d=11, sigmaColor=75, sigmaSpace=75)
        else:
            smoothed = cv2.GaussianBlur(enhanced_image, (9, 9), 0)

        # 2. GENERATE TIERS WITH POLYDP SPEEDUP
        step = 255 / max(2, num_tiers)
        thresholds = [int(step * i) for i in range(1, num_tiers)]

        for thresh in thresholds:
            _, binary = cv2.threshold(smoothed, thresh, 255, cv2.THRESH_BINARY)
            contours_topo, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours_topo:
                if cv2.arcLength(contour, closed=True) > self.min_contour_length_px:
                    # FASTER DRAWING: Approximate the contour
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, False)

                    points = approx_contour.squeeze().tolist()
                    if not isinstance(points, list) or not points: continue
                    if isinstance(points[0], int): points = [points]
                    path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
                    
                    if path:
                        all_paths_xy.append(path)
                        
            cv2.drawContours(preview, contours_topo, -1, 0, 1)

        # 3. INTERACTIVE EYE SOCKET DETECTION & ISOLATED FILL
        dark_mask = cv2.inRange(smoothed, 0, 45)
        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        eye_fill_mask = np.zeros_like(dark_mask)

        # Convert preview to BGR so we can draw colored interactive hints
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # Draw ALL candidate areas in light pink so the user knows what can be clicked
        cv2.drawContours(preview_bgr, dark_contours, -1, (200, 200, 255), 1)

        for contour in dark_contours:
            area = cv2.contourArea(contour)
            
            # Filter out microscopic noise so the user isn't clicking on 1-pixel dots
            if area > 15:
                # Check if any user point falls inside this contour
                is_selected = False
                for pt in user_eye_points:
                    # pointPolygonTest returns +1 if inside, 0 if exactly on the line
                    if cv2.pointPolygonTest(contour, (pt[0], pt[1]), False) >= 0:
                        is_selected = True
                        break
                        
                if is_selected:
                    cv2.drawContours(eye_fill_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Generate dense cross-hatching ONLY in the selected regions
        hatch_img = self._generate_dense_hatching(eye_fill_mask, spacing_px=3)
        
        h_contours, _ = cv2.findContours(hatch_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in h_contours:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if path:
                all_paths_xy.append(path)

        # Overlay the hatched fill on the preview in solid black
        preview_bgr[hatch_img == 255] = (0, 0, 0)

        # Draw blue dots exactly where the user clicked for visual feedback
        dot_radius = max(4, int(image_width * 0.005))
        for pt in user_eye_points:
            cv2.circle(preview_bgr, pt, dot_radius, (255, 0, 0), -1)

        if save_edge_path:
            try:
                cv2.imwrite(save_edge_path, preview_bgr)
            except Exception as e:
                logging.error(f"Failed to save preview: {e}")

        return all_paths_xy, image_width, image_height

    def _generate_dense_hatching(self, binary_mask, spacing_px):
        h, w = binary_mask.shape
        hatch_img = np.zeros_like(binary_mask)
        for y in range(0, h, spacing_px):
            cv2.line(hatch_img, (0, y), (w, y), 255, 1)
        for x in range(0, w, spacing_px):
            cv2.line(hatch_img, (x, 0), (x, h), 255, 1)
        return cv2.bitwise_and(hatch_img, binary_mask)