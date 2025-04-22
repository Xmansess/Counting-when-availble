# counting.py
from ultralytics import YOLO

import cv2 as cv
import numpy as np
import torch

import random
from tqdm import tqdm
import os

from drawing_bounds import detecting_area, draw_bounds

class DetectionModel:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Counting using device: {self.device}")
        self.detection_model = self.load_model(model_name)
        
    def load_model(self, model_name):
        model = YOLO(model_name)
        model.to(self.device)
        
        return model
    
    def __call__(self, frame, classes=18):
        return self.detection_model.track(frame, persist=True, verbose=False, classes=(classes))
    
class Counting_LiveStocks:
    def __init__(self, model_name, video_path, output_path=None):
        print(f"[INFO] Initializing counting for video: {video_path}")
        self.cap = cv.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_path}")
            return
            
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.process = tqdm(total=total_frames)
        
        frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        size = (frame_width, frame_height)
        
        # Fixed output path handling
        if output_path is None:
            output_folder = "./results/" + os.path.basename(video_path).split('.')[0]
        else:
            output_folder = output_path
            
        output_file = "analyzed_" + os.path.basename(video_path)
        output_path_ = os.path.join(output_folder, output_file)
        
        # Create directories if they don't exist
        os.makedirs(output_folder, exist_ok=True)
            
        print(f"[INFO] Output video will be saved to: {output_path_}")
        self.output = cv.VideoWriter(output_path_,
                                    cv.VideoWriter_fourcc(*'mp4v'),
                                    fps, size)
        
        self.detection_model = DetectionModel(model_name)
        
        self.id_color = {}
        
        self.font = cv.FONT_HERSHEY_SIMPLEX 
        self.org1 = (30, 35) 
        self.org2 = (30, 70) 
        self.fontScale = 1
        self.color = (0, 0, 255) 
        self.thickness = 1
        
    def plot_boxes(self, results, frame):
        if frame is None:
            return frame
            
        h, w, _ = frame.shape
        
        in_sight_count = 0
        for r in results:
            result = r.boxes.cpu()
            masks = r.masks
            object_ids = result.id
            
            if object_ids is not None:
                in_sight_count = len(object_ids)
                for i in range(in_sight_count):
                    r = random.randint(50,100)
                    g = random.randint(50,100)
                    b = random.randint(50,100)
                    object_id = object_ids[i].item()
                    if object_id not in self.id_color.keys():
                        self.id_color[object_id] = (b,g,r)
                        
                for i in range(in_sight_count):
                    b = result.xyxy[i]
                    object_id = object_ids[i].item()
                    
                    x1, x2 = int(b[0]), int(b[2])
                    y1, y2 = int(b[1]), int(b[3])
                    
                    # Check if masks exist and have the expected structure
                    if masks is not None and i < len(masks):
                        try:
                            object_mask = masks[i].data.cpu().numpy().astype('uint8')
                            
                            # Make sure coordinates are within bounds
                            x1 = max(0, min(x1, w-1))
                            x2 = max(0, min(x2, w-1))
                            y1 = max(0, min(y1, h-1))
                            y2 = max(0, min(y2, h-1))
                            
                            if x1 >= x2 or y1 >= y2:
                                continue  # Skip invalid boxes
                                
                            # Check mask shape and resize appropriately
                            if len(object_mask.shape) > 1:
                                object_mask_resize = cv.resize(object_mask[0], (w, h))
                                object_mask_resize = object_mask_resize[y1:y2, x1:x2]

                                # Create a color mask
                                detected_object = frame[y1:y2, x1:x2]
                                color_mask = np.zeros(detected_object.shape, dtype=np.uint8)
                                color_mask[object_mask_resize != 0] = self.id_color[object_id]
                                # Apply the color mask to the image
                                detected_object[object_mask_resize != 0] = 0.3*detected_object[object_mask_resize == 1] + 0.7*color_mask[object_mask_resize == 1]
                        except Exception as e:
                            print(f"[WARN] Error processing mask: {e}")
                    else:
                        # Just draw a box if no mask is available
                        cv.rectangle(frame, (x1, y1), (x2, y2), self.id_color[object_id], 2)
                
            frame = draw_bounds(frame)
            frame = cv.rectangle(frame, (5,5), (360,80), (238, 238, 175), -1)
            frame = cv.putText(frame, 'Quantity in sight:'+str(in_sight_count), self.org1, self.font,  
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
            frame = cv.putText(frame, 'Total:'+str(len(self.id_color.keys())), self.org2, self.font,  
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
        
        return frame

    def __call__(self):
        if not self.cap.isOpened():
            print("[ERROR] Video capture is not open")
            return
            
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            detecting_area_frame = detecting_area(frame)
            results = self.detection_model(detecting_area_frame)
            annotated_frame = self.plot_boxes(results, frame)
            self.output.write(annotated_frame)
            self.process.update(1)
            
        self.cap.release()
        self.output.release()
        print(f"[INFO] Video processing complete. Found {len(self.id_color.keys())} unique sheep.")
