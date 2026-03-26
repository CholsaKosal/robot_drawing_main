import cv2
import numpy as np
import logging
import mediapipe as mp
from smart_auto_eye_processor import SmartAutoEyeProcessor

class SmoothAutoEyeProcessor(SmartAutoEyeProcessor):
    """
    Mode 8: Enhanced Smooth Auto Eye + Denser Fill.
    A variation of Mode 6 focusing on extremely smooth contours (adjusted bilateral filters)
    and a much denser 4-way cross-hatch fill for the eyes.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, min_contour_length_px)

    def image_to_smooth_auto_tiers(self, image_path_or_array, num_tiers, filter_mode=2, save_edge_path=None, user_eye_points=None):
        if user_eye_points is None:
            user_eye_points = []

        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_path_or_array, np.ndarray):
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY) if len(image_path_or_array.shape) == 3 else image_path_or_array
        else:
            logging.error("Invalid input type")
            return None, 0, 0

        if image is None: return None, 0, 0

        image_height, image_width = image.shape[:2]
        all_paths_xy = []
        preview = np.full((image_height, image_width), 255, dtype=np.uint8)

        # 1. OPTIMIZE FOR SPEED & TOPOGRAPHY
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(image)

        # Mode 8 specific: highly smooth bilateral filters derived from the "medium edge pass"
        if filter_mode == 1:
            smoothed = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=70, sigmaSpace=70) # Light Smooth
        elif filter_mode == 2:
            smoothed = cv2.bilateralFilter(enhanced_image, d=13, sigmaColor=95, sigmaSpace=95) # Medium Smooth
        else:
            smoothed = cv2.bilateralFilter(enhanced_image, d=17, sigmaColor=130, sigmaSpace=130) # Heavy Smooth

        # 2. GENERATE TIERS
        step = 255 / max(2, num_tiers)
        thresholds = [int(step * i) for i in range(1, num_tiers)]

        for thresh in thresholds:
            _, binary = cv2.threshold(smoothed, thresh, 255, cv2.THRESH_BINARY)
            contours_topo, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours_topo:
                if cv2.arcLength(contour, closed=True) > self.min_contour_length_px:
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, False)

                    points = approx_contour.squeeze().tolist()
                    if not isinstance(points, list) or not points: continue
                    if isinstance(points[0], int): points = [points]
                    path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
                    
                    if path: all_paths_xy.append(path)
                    
            cv2.drawContours(preview, contours_topo, -1, 0, 1)

        # 3. EXACT 3D IRIS RECOGNITION (MediaPipe Tasks API)
        ai_eyes = []
        if self.detector:
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            try:
                results = self.detector.detect(mp_image)
                
                if results.face_landmarks:
                    for face_landmarks in results.face_landmarks:
                        iris_groups = [
                            [468, 469, 470, 471, 472], 
                            [473, 474, 475, 476, 477]
                        ]
                        
                        for iris_indices in iris_groups:
                            x_coords = []
                            y_coords = []
                            
                            for idx in iris_indices:
                                landmark = face_landmarks[idx]
                                x_coords.append(int(landmark.x * image_width))
                                y_coords.append(int(landmark.y * image_height))
                                
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            pad = 3
                            ex = max(0, x_min - pad)
                            ey = max(0, y_min - pad)
                            ew = min(image_width - ex, (x_max - x_min) + (pad * 2))
                            eh = min(image_height - ey, (y_max - y_min) + (pad * 2))
                            
                            if ew > 0 and eh > 0:
                                ai_eyes.append((ex, ey, ew, eh))
            except Exception as e:
                logging.error(f"Error during MediaPipe detection: {e}")

        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        eye_fill_mask = np.zeros_like(enhanced_image)

        # 4. LOCALIZED PUPIL EXTRACTION
        for (ex, ey, ew, eh) in ai_eyes:
            cv2.rectangle(preview_bgr, (ex, ey), (ex+ew, ey+eh), (200, 255, 200), 1)

            eye_roi = smoothed[ey:ey+eh, ex:ex+ew]
            min_val, _, _, _ = cv2.minMaxLoc(eye_roi)

            _, local_pupil_mask = cv2.threshold(eye_roi, min_val + 20, 255, cv2.THRESH_BINARY_INV)

            pupil_contours, _ = cv2.findContours(local_pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if pupil_contours:
                largest_pupil = max(pupil_contours, key=cv2.contourArea)
                largest_pupil += np.array([[ex, ey]])

                cv2.drawContours(eye_fill_mask, [largest_pupil], -1, 255, thickness=cv2.FILLED)

                M = cv2.moments(largest_pupil)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(preview_bgr, (cx, cy), max(3, int(image_width * 0.003)), (0, 200, 0), -1)

        # 5. USER INTERACTIVE FALLBACK
        if user_eye_points:
            global_dark_mask = cv2.inRange(smoothed, 0, 50)
            global_dark_contours, _ = cv2.findContours(global_dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in global_dark_contours:
                if cv2.contourArea(contour) > 10:
                    for pt in user_eye_points:
                        if cv2.pointPolygonTest(contour, (pt[0], pt[1]), False) >= 0:
                            cv2.drawContours(eye_fill_mask, [contour], -1, 255, thickness=cv2.FILLED)
                            break

        # 6. EXTREMELY DENSE HATCHING (Mode 8 Override)
        hatch_img = self._generate_dense_hatching(eye_fill_mask, spacing_px=2)
        h_contours, _ = cv2.findContours(hatch_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in h_contours:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if path: all_paths_xy.append(path)

        preview_bgr[hatch_img == 255] = (0, 0, 0)

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
        """
        Creates a massive 4-way cross-hatch for incredibly solid fills.
        """
        h, w = binary_mask.shape
        hatch_img = np.zeros_like(binary_mask)
        
        # Draw horizontal lines
        for y in range(0, h, spacing_px):
            cv2.line(hatch_img, (0, y), (w, y), 255, 1)
        # Draw vertical lines
        for x in range(0, w, spacing_px):
            cv2.line(hatch_img, (x, 0), (x, h), 255, 1)
        # Draw diagonal lines (\)
        for i in range(-h, w, spacing_px * 2):
            cv2.line(hatch_img, (i, 0), (i + h, h), 255, 1)
        # Draw diagonal lines (/)
        for i in range(0, w + h, spacing_px * 2):
            cv2.line(hatch_img, (i, 0), (i - h, h), 255, 1)
            
        return cv2.bitwise_and(hatch_img, binary_mask)