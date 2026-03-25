import cv2
import numpy as np
import logging
from classic_image_processor import ClassicImageProcessor

class AdvancedImageProcessor(ClassicImageProcessor):
    """
    Advanced processor that extracts classic contours AND generates 
    hatching (stripes) across the entire grayscale spectrum.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, min_contour_length_px)

    def image_to_contours_and_hatching(self, image_path_or_array, threshold1, threshold2, save_edge_path=None):
        """
        Combines Canny edge outlines with generated hatching paths for multiple intensity bands.
        """
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

        # 2. Get Classic Outlines (Canny) to preserve sharp details
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.arcLength(contour, closed=False) > self.min_contour_length_px:
                points = contour.squeeze().tolist()
                if not isinstance(points, list) or not points: continue
                if isinstance(points[0], int): points = [points]
                all_paths_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2])

        # 3. Generate Hatching Masks across the full spectrum
        # Format: (min_pixel_val, max_pixel_val, angle_degrees, spacing_pixels)
        shading_bands = [
            (0, 45, 45, 4),       # Tier 1: Deepest black - very tight spacing
            (46, 90, 135, 7),     # Tier 2: Dark gray - tight spacing, alternate angle
            (91, 140, 45, 11),    # Tier 3: Mid gray - medium spacing
            (141, 190, 135, 16),  # Tier 4: Light gray - wide spacing
            (191, 230, 45, 24)    # Tier 5: Very light gray - very wide spacing
            # 231 to 255 is considered white paper (no hatching)
        ]

        hatch_images = []

        for min_v, max_v, ang, space in shading_bands:
            # Create a mask for this specific range of pixels
            mask = cv2.inRange(blurred, min_v, max_v)
            
            # Generate the lines for this mask
            hatches = self._generate_hatching_lines(mask, angle_deg=ang, spacing_px=space)
            hatch_images.append(hatches)
            
            # Extract paths from the hatched masks
            h_contours, _ = cv2.findContours(hatches, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in h_contours:
                # No length filtering here so we don't lose small shading details
                points = contour.squeeze().tolist()
                if not isinstance(points, list) or not points: continue
                if isinstance(points[0], int): points = [points]
                path = [(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
                if path:
                    all_paths_xy.append(path)

        # 4. Save a preview image combining outlines and all shading layers
        if save_edge_path:
            preview = np.zeros((image_height, image_width), dtype=np.uint8)
            # Draw outlines
            cv2.drawContours(preview, contours, -1, 255, 1)
            # Add all hatching layers
            for h_img in hatch_images:
                preview = cv2.bitwise_or(preview, h_img)
                
            try:
                cv2.imwrite(save_edge_path, preview)
            except Exception as e:
                logging.error(f"Failed to save preview: {e}")

        return all_paths_xy, image_width, image_height

    def _generate_hatching_lines(self, binary_mask, angle_deg, spacing_px):
        """
        Draws parallel lines across an image and intersects them with the given mask.
        """
        h, w = binary_mask.shape
        hatch_img = np.zeros_like(binary_mask)
        diag = int(np.sqrt(h**2 + w**2))

        cx, cy = w // 2, h // 2
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for d in range(-diag, diag, spacing_px):
            px = cx - d * sin_a
            py = cy + d * cos_a

            p1x = int(px + diag * cos_a)
            p1y = int(py + diag * sin_a)
            p2x = int(px - diag * cos_a)
            p2y = int(py - diag * sin_a)

            cv2.line(hatch_img, (p1x, p1y), (p2x, p2y), 255, 1)

        # Intersect the generated lines with the target pixel mask
        return cv2.bitwise_and(hatch_img, binary_mask)