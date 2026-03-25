import cv2
import numpy as np
import logging
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from classic_image_processor import ClassicImageProcessor

class SmartAutoEyeProcessor(ClassicImageProcessor):
    """
    Mode 6: AI-Mapped Auto Eye Fill + Interactive.
    Upgraded to the modern MediaPipe Tasks API (compatible with Python 3.12+).
    Directly extracts exact Iris landmarks (ignores tilt, angle, and false positives).
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, max(10, min_contour_length_px // 2))
        
        # The new MediaPipe Tasks API requires a model file. 
        # We will automatically download it to the current directory if missing.
        self.model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(self.model_path):
            logging.info("Downloading MediaPipe Face Landmarker model (first time setup)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logging.info("MediaPipe model download complete.")
            except Exception as e:
                logging.error(f"Failed to download MediaPipe model: {e}")
        
        # Initialize the modern MediaPipe FaceLandmarker
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=10, # Support up to 10 people in a frame
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe FaceLandmarker: {e}")
            self.detector = None

    def image_to_smart_auto_tiers(self, image_path_or_array, num_tiers, filter_mode=2, save_edge_path=None, user_eye_points=None):
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
        
        if filter_mode == 1:
            smoothed = cv2.bilateralFilter(enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
        elif filter_mode == 2:
            smoothed = cv2.bilateralFilter(enhanced_image, d=11, sigmaColor=75, sigmaSpace=75)
        else:
            smoothed = cv2.GaussianBlur(enhanced_image, (9, 9), 0)

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
            # MediaPipe Tasks API requires an mp.Image object in RGB
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            try:
                results = self.detector.detect(mp_image)
                
                if results.face_landmarks:
                    for face_landmarks in results.face_landmarks:
                        # MediaPipe Landmark Indices for Irises:
                        # Left Iris: 468 (center), 469, 470, 471, 472 (edges)
                        # Right Iris: 473 (center), 474, 475, 476, 477 (edges)
                        iris_groups = [
                            [468, 469, 470, 471, 472], 
                            [473, 474, 475, 476, 477]
                        ]
                        
                        for iris_indices in iris_groups:
                            x_coords = []
                            y_coords = []
                            
                            # Convert normalized 3D coordinates to flat pixel coordinates
                            for idx in iris_indices:
                                landmark = face_landmarks[idx]
                                x_coords.append(int(landmark.x * image_width))
                                y_coords.append(int(landmark.y * image_height))
                                
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            # Create a tight bounding box around the exact Iris with a tiny padding
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
            # Draw light-green boxes to show exact AI Iris bounds
            cv2.rectangle(preview_bgr, (ex, ey), (ex+ew, ey+eh), (200, 255, 200), 1)

            # Isolate the precise iris box and threshold JUST this box
            eye_roi = smoothed[ey:ey+eh, ex:ex+ew]
            min_val, _, _, _ = cv2.minMaxLoc(eye_roi)

            # Threshold slightly above the absolute darkest point to capture the pupil
            _, local_pupil_mask = cv2.threshold(eye_roi, min_val + 20, 255, cv2.THRESH_BINARY_INV)

            pupil_contours, _ = cv2.findContours(local_pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if pupil_contours:
                # The pupil is the main dark mass in the isolated iris box
                largest_pupil = max(pupil_contours, key=cv2.contourArea)
                largest_pupil += np.array([[ex, ey]]) # Shift back to global coords

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

        # 6. HATCHING
        hatch_img = self._generate_dense_hatching(eye_fill_mask, spacing_px=3)
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
        h, w = binary_mask.shape
        hatch_img = np.zeros_like(binary_mask)
        for y in range(0, h, spacing_px):
            cv2.line(hatch_img, (0, y), (w, y), 255, 1)
        for x in range(0, w, spacing_px):
            cv2.line(hatch_img, (x, 0), (x, h), 255, 1)
        return cv2.bitwise_and(hatch_img, binary_mask)