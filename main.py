import os
import time
import cv2 as cv
from datetime import datetime

from detection import DetectionModel
from recorder import Recorder
from drawing_bounds import detecting_area

# === Configuration ===
STREAM_URL = "rtmp://192.168.180.237/live/streamkey"
MODEL_NAME = "yolov8x-seg.pt"
POLL_INTERVAL = 2  # seconds
OUTPUT_DIR = "./raw_clips"

# Ensure output folder exists
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try grab frame, with reconnect on failure
def grab_frame(cap, url):
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap = cv.VideoCapture(url)
        ret, frame = cap.read()
    return cap, frame

# === Main state machine ===
def main():
    print("[INFO] Starting main loop...")
    setup_output_dir()
    detector = DetectionModel(MODEL_NAME)
    recorder = Recorder(STREAM_URL)
    cap = cv.VideoCapture(STREAM_URL)
    state = "idle"

    try:
        while True:
            cap, frame = grab_frame(cap, STREAM_URL)
            if frame is None:
                print("[WARN] No frame grabbed, retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            frame_proc = detecting_area(frame)
            has_sheep = detector.detect(frame_proc)

            if has_sheep and state == "idle":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_fp = os.path.join(OUTPUT_DIR, f"clip_{ts}.mp4")
                print(f"[INFO] Sheep detected, starting recording to {out_fp}")
                recorder.start(out_fp)
                state = "recording"

            elif not has_sheep and state == "recording":
                print("[INFO] No sheep, stopping recording.")
                recorder.stop()
                state = "idle"

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, cleaning up...")
        if recorder.is_recording():
            recorder.stop()
        cap.release()

if __name__ == "__main__":
    main()
