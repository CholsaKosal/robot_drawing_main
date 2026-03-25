import cv2
import numpy as np
import logging
from classic_image_processor import ClassicImageProcessor

class SharpDetailProcessor(ClassicImageProcessor):
    """
    Mode 4: Portrait Detail + Solid Fill.
    Uses CLAHE and Median Blur for sharp edges, but ALSO detects the darkest
    regions (like pupils and dark hair) and generates dense hatching to fill
    them in so they don't look "hollow".
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        # Halve the minimum contour length to catch short eyelash/hair strokes
        super().__init__(a4_width_mm, a4_height_mm, max(10, min_contour_length_px // 2))

    def image_to_contours(self, image_path_or_array, threshold1, threshold2, save_edge_path=None):
        """
        Convert image to contours and fill in the dark voids.
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

        all_paths_xy = []

        # 1. OPTIMIZE IMAGE (CLAHE + Median Blur)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast_img = clahe.apply(image)
        denoised_img = cv2.medianBlur(contrast_img, 5)

        # 2. EXTRACT OUTLINES (Canny Edge Detection)
        edges = cv2.Canny(denoised_img, threshold1, threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.arcLength(contour, closed=False) > self.min_contour_length_px:
                points = contour.squeeze().tolist()
                if not isinstance(points, list) or not points: continue
                if isinstance(points[0], int): points = [points]
                path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
                if path:
                    all_paths_xy.append(path)

        # 3. THE FIX: SOLID FILL FOR "HOLLOW" AREAS
        # Isolate the darkest pixels in the image (0 = pure black, 50 = dark gray)
        # Anything in this range will get colored in.
        dark_mask = cv2.inRange(denoised_img, 0, 50)
        
        # Generate tight cross-hatching lines inside those dark areas
        # Spacing of 3 pixels usually gives a solid color look with a 0.5mm pen
        hatch_img = self._generate_dense_hatching(dark_mask, spacing_px=3)
        
        # Extract paths from the hatched fill
        h_contours, _ = cv2.findContours(hatch_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in h_contours:
            # No length filtering here so we don't lose the tiny fill lines inside small pupils
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if path:
                all_paths_xy.append(path)

        # 4. GENERATE PREVIEW
        if save_edge_path:
            # Start with a white canvas
            preview = np.full((image_height, image_width), 255, dtype=np.uint8)
            # Draw outlines in black
            cv2.drawContours(preview, contours, -1, 0, 1) 
            # Overlay the hatching fill in black
            preview[hatch_img == 255] = 0
            
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