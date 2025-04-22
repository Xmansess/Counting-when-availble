# detection.py
from ultralytics import YOLO
import torch
import time
import numpy as np

class DetectionModel:
    def __init__(self, model_name: str):
        # Load YOLO once, keep on chosen device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        self.model = self.load_model(model_name)
        
    def load_model(self, model_name):
        print(f"[INFO] Loading model: {model_name}")
        start_time = time.time()
        model = YOLO(model_name)
        model.to(self.device)
        print(f"[INFO] Model loaded in {time.time() - start_time:.2f} seconds")
        return model
    
    def is_black_frame(self, frame, threshold=10):
        # Check if frame is almost completely black
        gray = np.mean(frame)
        is_black = gray < threshold
        if is_black:
            print(f"[INFO] Frame appears to be black (mean value: {gray:.2f})")
        return is_black
    
    def detect(self, frame, classes: int = 18, conf_threshold: float = 0.35) -> bool:
        # First check if this is just a black frame
        if self.is_black_frame(frame):
            return False
        
        # Returns True if any sheep (class=18) detected in the frame with confidence > threshold
        start_time = time.time()
        results = self.model.track(frame, persist=False, verbose=False, classes=(classes,), conf=conf_threshold)
        
        sheep_detected = False
        for r in results:
            if len(r.boxes) > 0:
                # Check confidence scores
                conf_scores = r.boxes.conf.cpu().numpy()
                high_conf_boxes = conf_scores > conf_threshold
                sheep_count = np.sum(high_conf_boxes)
                
                if sheep_count > 0:
                    sheep_detected = True
                    print(f"[INFO] Detection took {time.time() - start_time:.2f} seconds, found {sheep_count} sheep with conf > {conf_threshold}")
                else:
                    print(f"[INFO] Found potential sheep but confidence too low (max: {np.max(conf_scores):.2f})")
        
        if not sheep_detected:
            print(f"[INFO] Detection took {time.time() - start_time:.2f} seconds, no sheep found")
        
        return sheep_detected
