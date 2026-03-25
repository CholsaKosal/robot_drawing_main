import logging
from smart_auto_eye_processor import SmartAutoEyeProcessor

class RealImageDrawingProcessor(SmartAutoEyeProcessor):
    """
    Mode 7: Real Image Drawing Shortcut.
    A direct 1-click shortcut to Mode 6, hardcoded to 60 tiers 
    and Medium Detail Edge Pass (filter_mode=2) for optimal real-image contours.
    """
    def __init__(self, a4_width_mm: float, a4_height_mm: float, min_contour_length_px: int):
        super().__init__(a4_width_mm, a4_height_mm, min_contour_length_px)

    def image_to_real_image_drawing(self, image_path_or_array, save_edge_path=None, user_eye_points=None):
        """
        Bypasses threshold inputs and routes directly to the Smart Auto-Tier logic 
        using the optimal 60-tier, medium-pass configuration.
        """
        logging.info("Processing Mode 7: Hardcoded to 60 Tiers, Medium Edge Pass")
        return self.image_to_smart_auto_tiers(
            image_path_or_array=image_path_or_array,
            num_tiers=60,
            filter_mode=2, # Medium Pass
            save_edge_path=save_edge_path,
            user_eye_points=user_eye_points
        )