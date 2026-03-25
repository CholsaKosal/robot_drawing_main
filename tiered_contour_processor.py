import cv2
import numpy as np
import logging
from classic_image_processor import ClassicImageProcessor

class TieredContourProcessor(ClassicImageProcessor):
    """
    Mode 3: Pure Topographical Contours + Dark Area Fill.
    Uses Edge-Preserving Blurs (Bilateral Filtering) to create smooth volume 
    on gradients (skin) while tightly bunching contour lines around sharp details.
    NOW INCLUDES: A solid-fill generator to shade in hollow eyes and dark hair.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        # We halve the minimum contour length specifically for this processor 
        # so we don't lose the tiny topographical rings that form pupils and highlights.
        super().__init__(a4_width_mm, a4_height_mm, max(10, min_contour_length_px // 2))

    def image_to_tiered_contours(self, image_path_or_array, num_tiers, filter_mode=2, save_edge_path=None):
        """
        Slices the image into topographical lines and fills in the deepest voids.
        """
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
        
        # White canvas for our preview
        preview = np.full((image_height, image_width), 255, dtype=np.uint8)

        # 1. Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(image)

        # 2. Smart Blurring (The magic that preserves edges while smoothing skin)
        if filter_mode == 1:
            smoothed = cv2.bilateralFilter(enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
        elif filter_mode == 2:
            smoothed = cv2.bilateralFilter(enhanced_image, d=11, sigmaColor=75, sigmaSpace=75)
        else:
            smoothed = cv2.GaussianBlur(enhanced_image, (9, 9), 0)

        # 3. Topographical Slicing
        step = 255 / max(2, num_tiers)
        thresholds = [int(step * i) for i in range(1, num_tiers)]

        for thresh in thresholds:
            _, binary = cv2.threshold(smoothed, thresh, 255, cv2.THRESH_BINARY)
            contours_topo, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours_topo:
                if cv2.arcLength(contour, closed=True) > self.min_contour_length_px:
                    points = contour.squeeze().tolist()
                    if not isinstance(points, list) or not points: continue
                    if isinstance(points[0], int): points = [points]
                    path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
                    
                    if path:
                        all_paths_xy.append(path)
                        
            # Draw topo lines on preview (black)
            cv2.drawContours(preview, contours_topo, -1, 0, 1)

        # 4. THE FIX: SOLID FILL FOR HOLLOW AREAS (Eyes, Dark Hair)
        # Isolate the darkest pixels (0 to 45). Adjust 45 up or down if you want more/less fill.
        dark_mask = cv2.inRange(smoothed, 0, 45)
        
        # Generate tight cross-hatching (3px spacing) inside those dark areas
        hatch_img = self._generate_dense_hatching(dark_mask, spacing_px=3)
        
        # Extract paths from the hatched fill
        h_contours, _ = cv2.findContours(hatch_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in h_contours:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if path:
                all_paths_xy.append(path)

        # Overlay the hatching fill onto the preview
        preview[hatch_img == 255] = 0

        # Save Preview
        if save_edge_path:
            try:
                cv2.imwrite(save_edge_path, preview)
            except Exception as e:
                logging.error(f"Failed to save preview: {e}")

        return all_paths_xy, image_width, image_height

    def _generate_dense_hatching(self, binary_mask, spacing_px):
        """
        Creates a tight grid of cross-hatching over the image, 
        then crops it down to only exist inside the target binary mask.
        """
        h, w = binary_mask.shape
        hatch_img = np.zeros_like(binary_mask)
        
        # Draw horizontal lines tightly packed
        for y in range(0, h, spacing_px):
            cv2.line(hatch_img, (0, y), (w, y), 255, 1)
            
        # Draw vertical lines tightly packed to create a grid/solid fill
        for x in range(0, w, spacing_px):
            cv2.line(hatch_img, (x, 0), (x, h), 255, 1)
            
        # Keep only the lines that fall inside the dark areas
        return cv2.bitwise_and(hatch_img, binary_mask)