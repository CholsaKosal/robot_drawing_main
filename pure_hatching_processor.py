import cv2
import numpy as np
import logging
import mediapipe as mp
from advanced_image_processor import AdvancedImageProcessor

class PureHatchingProcessor(AdvancedImageProcessor):
    """
    Mode 9: Pure Hatching.
    Everything from Mode 2 (Advanced), but without Canny outlines. Just pure hatching.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, min_contour_length_px)

    def image_to_pure_hatching(self, image_path_or_array, threshold1, threshold2, num_tiers, save_edge_path=None, user_eye_points=None, invert_density=False):
        if user_eye_points is None:
            user_eye_points = []

        # 1. Load Image
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

        # 2. Compute Edges purely for the safe_mask (DO NOT append to all_paths_xy)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)

        # 3. Generate Hatching Paths based on dynamic num_tiers
        dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        safe_mask = cv2.bitwise_not(dilated_edges)

        shading_bands = []
        step = 255 / max(2, num_tiers)
        num_shading_tiers = num_tiers - 1
        
        for i in range(num_shading_tiers): 
            min_v = int(i * step)
            max_v = int((i + 1) * step) - 1
            ang = 45 if i % 2 == 0 else 135
            
            # Switch logic based on toggle state
            space_idx = (num_shading_tiers - 1 - i) if invert_density else i
            space = max(2, 4 + (space_idx * 3))
            
            shading_bands.append((min_v, max_v, ang, space))

        hatch_preview_layer = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Calculate a healthy minimum length for body/background shading lines
        min_hatch_len = max(15, self.min_contour_length_px // 10)

        for min_v, max_v, ang, space in shading_bands:
            mask = cv2.inRange(blurred, min_v, max_v)
            mask = cv2.bitwise_and(mask, safe_mask) 
            
            paths = self._generate_hatching_paths(mask, ang, space, min_length_px=min_hatch_len)
            if paths:
                all_paths_xy.extend(paths)
                for path in paths:
                    cv2.line(hatch_preview_layer, path[0], path[-1], 255, 1)

        # 4. --- AI PUPIL DETECTION & INTERACTIVE FILL (WITH CACHE) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(image)

        img_sum = int(image.sum()) if image.size > 0 else 0
        img_hash = hash((image.shape, img_sum))

        with self._cache_lock:
            if self._last_image_hash == img_hash:
                ai_eyes = self._last_ai_eyes
            else:
                ai_eyes = []
                if self.detector:
                    rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                    try:
                        results = self.detector.detect(mp_image)
                        if results.face_landmarks:
                            for face_landmarks in results.face_landmarks:
                                iris_groups = [[468, 469, 470, 471, 472], [473, 474, 475, 476, 477]]
                                for iris_indices in iris_groups:
                                    x_coords = [int(face_landmarks[idx].x * image_width) for idx in iris_indices]
                                    y_coords = [int(face_landmarks[idx].y * image_height) for idx in iris_indices]
                                    x_min, x_max = min(x_coords), max(x_coords)
                                    y_min, y_max = min(y_coords), max(y_coords)
                                    pad = 3
                                    ex, ey = max(0, x_min - pad), max(0, y_min - pad)
                                    ew = min(image_width - ex, (x_max - x_min) + (pad * 2))
                                    eh = min(image_height - ey, (y_max - y_min) + (pad * 2))
                                    if ew > 0 and eh > 0:
                                        ai_eyes.append((ex, ey, ew, eh))
                    except Exception as e:
                        logging.error(f"Error during MediaPipe detection: {e}")
                self._last_ai_eyes = ai_eyes
                self._last_image_hash = img_hash

        eye_fill_mask = np.zeros_like(enhanced_image)

        # AI Found Eyes -> Extract darkest part (Pupil)
        for (ex, ey, ew, eh) in ai_eyes:
            eye_roi = blurred[ey:ey+eh, ex:ex+ew]
            min_val, _, _, _ = cv2.minMaxLoc(eye_roi)
            _, local_pupil_mask = cv2.threshold(eye_roi, min_val + 20, 255, cv2.THRESH_BINARY_INV)
            pupil_contours, _ = cv2.findContours(local_pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if pupil_contours:
                largest_pupil = max(pupil_contours, key=cv2.contourArea)
                largest_pupil += np.array([[ex, ey]])
                cv2.drawContours(eye_fill_mask, [largest_pupil], -1, 255, thickness=cv2.FILLED)

        # Interactive User Clicks -> Isolate clicked blob
        if user_eye_points:
            global_dark_mask = cv2.inRange(blurred, 0, 50)
            global_dark_contours, _ = cv2.findContours(global_dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in global_dark_contours:
                if cv2.contourArea(contour) > 10:
                    for pt in user_eye_points:
                        if cv2.pointPolygonTest(contour, (pt[0], pt[1]), False) >= 0:
                            cv2.drawContours(eye_fill_mask, [contour], -1, 255, thickness=cv2.FILLED)
                            break

        # Generate very dense mathematical cross-hatching inside the detected/clicked eyes
        eye_paths = []
        eye_paths.extend(self._generate_hatching_paths(eye_fill_mask, 45, 2, min_length_px=4))
        eye_paths.extend(self._generate_hatching_paths(eye_fill_mask, 135, 2, min_length_px=4))
        
        if eye_paths:
            all_paths_xy.extend(eye_paths)
            for path in eye_paths:
                cv2.line(hatch_preview_layer, path[0], path[-1], 255, 1)

        # 5. Save a preview image combining ONLY shading and UI overlays (no outlines)
        if save_edge_path:
            preview = hatch_preview_layer
            
            # Convert to color to draw visual feedback boxes and dots
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            for (ex, ey, ew, eh) in ai_eyes:
                cv2.rectangle(preview_bgr, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            
            dot_radius = max(4, int(image_width * 0.005))
            for pt in user_eye_points:
                cv2.circle(preview_bgr, pt, dot_radius, (255, 0, 0), -1)

            try:
                cv2.imwrite(save_edge_path, preview_bgr)
            except Exception as e:
                logging.error(f"Failed to save preview: {e}")

        return all_paths_xy, image_width, image_height