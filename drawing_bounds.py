import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Define points for the ROI polygon. It's good practice to define them once.
ROI_POINTS = np.array([[0,650], [600,500], [700,400], [1400,450], [1300,1050], [0,1050]])

def detecting_area(frame):
    if frame is None:
        logger.warning("detecting_area received a None frame.")
        return None # Or raise an error, depending on desired behavior

    logger.debug(f"Applying detecting_area mask to frame with shape: {frame.shape}")
    try:
        # h, w, _ = frame.shape # Not strictly needed if ROI_POINTS are absolute
        mask = np.zeros_like(frame)
        cv.fillPoly(mask, [ROI_POINTS], (1,1,1)) # Use the defined ROI_POINTS
        masked_frame = frame * mask
        logger.debug("detecting_area mask applied successfully.")
        return masked_frame
    except Exception as e:
        logger.error(f"Error in detecting_area: {e}", exc_info=True)
        return frame # Return original frame on error to prevent crash, or handle differently


def draw_bounds(frame):
    if frame is None:
        logger.warning("draw_bounds received a None frame.")
        return None

    logger.debug(f"Drawing bounds on frame with shape: {frame.shape}")
    try:
        # ROI lines color (example, can be parameterized)
        # line_color = (0, 255, 255) # Yellow, as in original
        # cv.polylines(frame, [ROI_POINTS], isClosed=True, color=line_color, thickness=2) # Example of drawing lines

        # Shading inside the ROI
        # Create a single channel mask for shading
        shading_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv.fillPoly(shading_mask, [ROI_POINTS], 1) # Use the defined ROI_POINTS

        overlay = np.zeros_like(frame)
        overlay_color = (221, 218, 250) # Light lavender color from original
        overlay[shading_mask == 1] = overlay_color

        # Apply weighted sum for shading effect
        # Ensure frame is of a type that supports floating point for alpha blending
        if not np.issubdtype(frame.dtype, np.floating):
            frame_float = frame.astype(np.float32)
        else:
            frame_float = frame.copy() # Use a copy to avoid modifying original if it's already float

        frame_float[shading_mask == 1] = 0.7 * frame_float[shading_mask == 1] + 0.3 * overlay[shading_mask == 1]

        # Convert back to uint8 if original was uint8
        if np.issubdtype(frame.dtype, np.integer):
            shaded_frame = frame_float.astype(np.uint8)
        else:
            shaded_frame = frame_float # Keep as float if original was float

        logger.debug("Bounds drawn and area shaded successfully.")
        return shaded_frame
    except Exception as e:
        logger.error(f"Error in draw_bounds: {e}", exc_info=True)
        return frame # Return original frame on error
