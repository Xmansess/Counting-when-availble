from ultralytics import YOLO
import torch

class DetectionModel:
    def __init__(self, model_name: str):
        # Load YOLO once, keep on chosen device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, frame, classes: int = 18) -> bool:
        # Returns True if any sheep (class=18) detected in the frame
        results = self.model.track(frame, persist=False, verbose=False, classes=(classes,))
        for r in results:
            if len(r.boxes) > 0:
                return True
        return False
