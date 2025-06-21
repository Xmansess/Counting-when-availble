# detection.py
from ultralytics import YOLO
import torch
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DetectionModel:
    def __init__(self, model_name: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing DetectionModel using device: {self.device}")
        try:
            self.model = self.load_model(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize DetectionModel: {e}", exc_info=True)
            raise # Re-raise the exception to signal failure to the caller
        
    def load_model(self, model_name):
        logger.info(f"Loading YOLO model: {model_name} onto device: {self.device}")
        start_time = time.time()
        try:
            model = YOLO(model_name)
            model.to(self.device)
            end_time = time.time()
            logger.info(f"Model {model_name} loaded successfully in {end_time - start_time:.2f} seconds")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            raise # Re-raise to indicate that model loading failed

    def detect(self, frame, classes: int = 18, conf_threshold: float = 0.5) -> bool:
        if frame is None:
            logger.warning("Detection attempt on a None frame.")
            return False

        logger.debug(f"Performing detection on frame with shape: {frame.shape}, classes: {classes}, conf_threshold: {conf_threshold}")
        start_time = time.time()
        
        try:
            results = self.model.track(frame, persist=False, verbose=False, classes=(classes,), conf=conf_threshold)
        except Exception as e:
            logger.error(f"Exception during model.track: {e}", exc_info=True)
            return False # Return False on error to prevent downstream issues

        sheep_detected = False
        detected_sheep_count = 0 # Initialize count for this detection call

        if results: # Check if results is not None or empty
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    conf_scores = r.boxes.conf.cpu().numpy()
                    # Filter boxes by confidence threshold
                    high_conf_indices = conf_scores >= conf_threshold # Ensure comparison is correct

                    # Count how many detections meet the threshold
                    current_sheep_in_result = np.sum(high_conf_indices)

                    if current_sheep_in_result > 0:
                        detected_sheep_count += current_sheep_in_result # Accumulate count
                        sheep_detected = True # Set to True if any sheep are detected meeting criteria
                        logger.debug(f"Detected {current_sheep_in_result} sheep in a result object with conf > {conf_threshold}. Max conf: {np.max(conf_scores):.2f}")
                    else:
                        # This case means boxes were detected, but none met the confidence threshold
                        if len(conf_scores)>0 : # Check if there were any scores to report max from
                             logger.debug(f"Potential sheep found but confidence too low (max: {np.max(conf_scores):.2f}, threshold: {conf_threshold})")
                        else:
                             logger.debug("No boxes returned confidence scores.")
                else:
                    logger.debug("A result object had no boxes or boxes attribute was None.")
        else:
            logger.debug("Model produced no results.")

        detection_time = time.time() - start_time
        if sheep_detected:
            logger.info(f"Detection completed in {detection_time:.2f}s. Found {detected_sheep_count} sheep meeting criteria.")
        else:
            logger.info(f"Detection completed in {detection_time:.2f}s. No sheep found meeting criteria.")
        
        return sheep_detected
