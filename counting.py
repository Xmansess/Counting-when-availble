# counting.py
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import random
from tqdm import tqdm
import os
import logging

from drawing_bounds import detecting_area, draw_bounds

logger = logging.getLogger(__name__)

# This DetectionModel is specific to counting.py for tracking.
# It's different from the one in detection.py which is for simple detection.
class DetectionModelForCounting:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Counting_LiveStocks.DetectionModel: Using device: {self.device}")
        try:
            self.detection_model = self.load_model(model_name)
        except Exception as e:
            logger.error(f"Counting_LiveStocks.DetectionModel: Failed to load model {model_name}: {e}", exc_info=True)
            raise
        
    def load_model(self, model_name):
        logger.info(f"Counting_LiveStocks.DetectionModel: Loading model {model_name}")
        model = YOLO(model_name)
        model.to(self.device)
        logger.info(f"Counting_LiveStocks.DetectionModel: Model {model_name} loaded successfully.")
        return model
    
    def __call__(self, frame, classes=18):
        logger.debug("Counting_LiveStocks.DetectionModel: Performing tracking.")
        # Add persist=True for tracking, verbose=False to reduce console spam from YOLO
        return self.detection_model.track(frame, persist=True, verbose=False, classes=(classes,))
    
class Counting_LiveStocks:
    def __init__(self, model_name, video_path, output_path=None):
        logger.info(f"Initializing Counting_LiveStocks for video: {video_path}")
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            # Consider raising an exception here to signal failure
            raise IOError(f"Failed to open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties: Total frames: {self.total_frames}")
        self.process = tqdm(total=self.total_frames, desc=f"Counting in {os.path.basename(video_path)}")
        
        frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        size = (frame_width, frame_height)
        logger.info(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
        
        if output_path is None:
            # Ensure "./results/" exists or is created safely
            default_results_dir = "./results/"
            os.makedirs(default_results_dir, exist_ok=True)
            output_folder = os.path.join(default_results_dir, os.path.basename(video_path).split('.')[0])
        else:
            output_folder = output_path # Assumes output_path is a directory
            
        # Ensure the specific output_folder for this video exists
        os.makedirs(output_folder, exist_ok=True)
            
        output_file_name = "analyzed_" + os.path.basename(video_path)
        self.output_video_path = os.path.join(output_folder, output_file_name)
        
        logger.info(f"Output video will be saved to: {self.output_video_path}")
        try:
            self.output = cv.VideoWriter(self.output_video_path,
                                        cv.VideoWriter_fourcc(*'mp4v'),
                                        fps, size)
        except Exception as e:
            logger.error(f"Failed to create VideoWriter for {self.output_video_path}: {e}", exc_info=True)
            self.cap.release() # Release input video capture
            raise # Re-raise to signal failure

        try:
            self.detection_model = DetectionModelForCounting(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize internal detection model for counting: {e}", exc_info=True)
            self.cap.release()
            if hasattr(self, 'output') and self.output.isOpened():
                 self.output.release()
            raise

        self.id_color = {}
        self.font = cv.FONT_HERSHEY_SIMPLEX 
        self.org1 = (30, 35) 
        self.org2 = (30, 70) 
        self.fontScale = 1
        self.color = (0, 0, 255) 
        self.thickness = 1
        self.current_frame_number = 0
        
    def plot_boxes(self, results, frame):
        if frame is None:
            logger.warning("plot_boxes received a None frame.")
            return None # Or an empty frame of the correct size if that's more appropriate
            
        h, w, _ = frame.shape
        in_sight_count = 0

        if results: # Ensure results is not None or empty
            for r_idx, r in enumerate(results): # Enumerate for detailed logging if needed
                if r.boxes is None or r.boxes.id is None:
                    logger.debug(f"Result object {r_idx} has no boxes or no IDs, skipping.")
                    continue

                result_boxes = r.boxes.cpu() # Get boxes once
                object_ids = result_boxes.id.numpy() # Get IDs as numpy array

                in_sight_count = len(object_ids)
                logger.debug(f"Frame {self.current_frame_number}: {in_sight_count} objects in sight in this result.")

                for i in range(in_sight_count):
                    object_id = int(object_ids[i]) # Ensure it's a Python int for dict keys
                    if object_id not in self.id_color:
                        rand_color = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
                        self.id_color[object_id] = rand_color
                        logger.info(f"Frame {self.current_frame_number}: New unique ID {object_id} detected. Total unique: {len(self.id_color)}")
                        
                for i in range(in_sight_count):
                    box_coords = result_boxes.xyxy[i].numpy()
                    object_id = int(object_ids[i])
                    
                    x1, y1, x2, y2 = map(int, box_coords)
                    
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                            
                    if x1 >= x2 or y1 >= y2:
                        logger.warning(f"Frame {self.current_frame_number}: Invalid box coordinates for ID {object_id} after clamping: ({x1},{y1})-({x2},{y2}). Skipping.")
                        continue
                                
                    current_color = self.id_color.get(object_id, (255,0,0)) # Default color if somehow not set

                    # Mask processing
                    if r.masks is not None and i < len(r.masks):
                        try:
                            mask_data = r.masks[i].data.cpu().numpy().astype('uint8')
                            if len(mask_data.shape) > 1 and mask_data.shape[0] > 0: # Check if mask_data[0] is valid
                                # Resize the first channel of the mask to frame dimensions
                                object_mask_resized_full = cv.resize(mask_data[0], (w, h))
                                # Crop the resized mask to the bounding box area
                                object_mask_cropped = object_mask_resized_full[y1:y2, x1:x2]

                                detected_object_roi = frame[y1:y2, x1:x2]
                                color_mask_overlay = np.zeros_like(detected_object_roi, dtype=np.uint8)
                                color_mask_overlay[object_mask_cropped != 0] = current_color

                                # Apply blending
                                blended_roi = cv.addWeighted(detected_object_roi, 0.7, color_mask_overlay, 0.3, 0)
                                frame[y1:y2, x1:x2] = blended_roi
                            else:
                                logger.debug(f"Frame {self.current_frame_number}: Mask data for ID {object_id} has unexpected shape {mask_data.shape}. Drawing bounding box instead.")
                                cv.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
                        except Exception as e:
                            logger.warning(f"Frame {self.current_frame_number}: Error processing mask for ID {object_id}: {e}. Drawing bounding box instead.", exc_info=True)
                            cv.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
                    else:
                        logger.debug(f"Frame {self.current_frame_number}: No mask available for ID {object_id} or index out of bounds. Drawing bounding box.")
                        cv.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
        else:
            logger.debug(f"Frame {self.current_frame_number}: No results from tracker for this frame.")
                
        try:
            frame = draw_bounds(frame) # Assuming this is for ROI visualization
            # Overlay for text
            cv.rectangle(frame, (5,5), (360,80), (238, 238, 175), -1) # Background for text
            cv.putText(frame, f'Quantity in sight: {in_sight_count}', self.org1, self.font,
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
            cv.putText(frame, f'Total unique: {len(self.id_color)}', self.org2, self.font,
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
        except Exception as e:
            logger.error(f"Frame {self.current_frame_number}: Error drawing bounds or text overlay: {e}", exc_info=True)
        
        return frame

    def __call__(self):
        if not self.cap.isOpened():
            logger.error(f"Video capture for {self.video_path} is not open. Aborting processing.")
            return # Already logged in init, but double check

        logger.info(f"Starting video processing loop for {self.video_path}")
        self.current_frame_number = 0
            
        try:
            while self.cap.isOpened():
                self.current_frame_number += 1
                success, frame = self.cap.read()
                if not success:
                    logger.info(f"End of video {self.video_path} or failed to read frame at position {self.current_frame_number}.")
                    break

                if self.current_frame_number % 100 == 0: # Log progress every 100 frames
                    logger.info(f"Processing frame {self.current_frame_number}/{self.total_frames} of {self.video_path}")

                detecting_area_frame = detecting_area(frame.copy()) # Use a copy if detecting_area modifies it

                # Perform detection/tracking
                results = self.detection_model(detecting_area_frame)

                # Annotate frame
                annotated_frame = self.plot_boxes(results, frame) # Pass original frame for annotation

                if annotated_frame is not None:
                    self.output.write(annotated_frame)
                else:
                    logger.warning(f"Frame {self.current_frame_number}: Annotated frame is None, not writing to output.")
                    # Optionally write the original frame or a black frame
                    # self.output.write(frame)

                self.process.update(1)
        except Exception as e:
            logger.error(f"Error during video processing loop for {self.video_path} at frame {self.current_frame_number}: {e}", exc_info=True)
        finally:
            logger.info(f"Releasing video capture and writer for {self.video_path}.")
            self.cap.release()
            if hasattr(self, 'output') and self.output.isOpened(): # Ensure output was created and is open
                self.output.release()
            self.process.close()
            logger.info(f"Video processing complete for {self.video_path}. Found {len(self.id_color.keys())} unique sheep.")
            # For main.py to pick up the count
            self.final_sheep_count = len(self.id_color.keys())
