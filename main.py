# main.py
import os
import time
import cv2 as cv
from datetime import datetime
import subprocess

from detection import DetectionModel
from recorder import Recorder
from drawing_bounds import detecting_area
from counting import Counting_LiveStocks

# === Configuration ===
STREAM_URL = "rtmp://192.168.180.237/live/streamkey"
# Ավելի թեթև մոդել
MODEL_NAME = "yolov8n-seg.pt"  # փոխարինել ծանր x մոդելը թեթև n մոդելով
POLL_INTERVAL = 2  # seconds
OUTPUT_DIR = "./raw_clips"
ANALYSIS_DIR = "./analyzed_clips"

# Parameters to make the system more stable
NO_SHEEP_THRESHOLD = 5  # How many intervals to wait before stopping recording
MIN_RECORDING_TIME = 30  # Minimum recording time in seconds

# Ensure output folders exist
def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Try grab frame, with reconnect on failure
def grab_frame(cap, url):
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame, attempting reconnection...")
        cap.release()
        time.sleep(1)  # Մի փոքր սպասել մինչև կրկին միանալը
        cap = cv.VideoCapture(url)
        ret, frame = cap.read()
    return cap, frame if ret else (cap, None)

# Run the counting analysis on a completed video
def analyze_video(video_path):
    print(f"[INFO] Analyzing video: {video_path}")
    counter = Counting_LiveStocks(MODEL_NAME, video_path, ANALYSIS_DIR)
    counter()
    sheep_count = len(counter.id_color.keys())
    print(f"[INFO] Analysis complete. Counted {sheep_count} unique sheep.")
    return sheep_count

# === Main state machine ===
def main():
    print("[INFO] Starting main loop...")
    setup_dirs()
    detector = DetectionModel(MODEL_NAME)
    recorder = Recorder(STREAM_URL)
    cap = cv.VideoCapture(STREAM_URL)
    
    # Check if stream is opened successfully
    if not cap.isOpened():
        print("[ERROR] Failed to open stream. Check URL and connection.")
        return
    
    state = "idle"
    
    # Stabilization variables
    no_sheep_count = 0
    recording_start_time = 0
    current_recording_path = None

    try:
        while True:
            cap, frame = grab_frame(cap, STREAM_URL)
            if frame is None:
                print("[WARN] No frame grabbed, retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            print("[INFO] Processing frame...")  # Debug info
            frame_proc = detecting_area(frame)
            # main.py-ի փոքր փոփոխություն (միայն համապատասխան մասը)
            has_sheep = detector.detect(frame_proc, conf_threshold=0.5)
            print(f"[INFO] Sheep detected: {has_sheep}")  # Debug info

            # State machine
            if has_sheep and state == "idle":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_recording_path = os.path.join(OUTPUT_DIR, f"clip_{ts}.mp4")
                print(f"[INFO] Sheep detected, starting recording to {current_recording_path}")
                recorder.start(current_recording_path)
                recording_start_time = time.time()
                state = "recording"
                no_sheep_count = 0

            elif not has_sheep and state == "recording":
                no_sheep_count += 1
                current_duration = time.time() - recording_start_time
                
                if no_sheep_count >= NO_SHEEP_THRESHOLD and current_duration >= MIN_RECORDING_TIME:
                    print(f"[INFO] No sheep for {no_sheep_count * POLL_INTERVAL}s, stopping recording.")
                    recorder.stop()
                    state = "idle"
                    no_sheep_count = 0
                    
                    # Run analysis on the completed video
                    if current_recording_path and os.path.exists(current_recording_path):
                        sheep_count = analyze_video(current_recording_path)
                        print(f"[RESULT] Detected {sheep_count} unique sheep in the video.")
                    else:
                        print("[WARN] Recording file not found for analysis.")
                else:
                    if no_sheep_count > 0:
                        print(f"[INFO] No sheep detected for {no_sheep_count * POLL_INTERVAL}s, but continuing recording.")

            elif has_sheep and state == "recording":
                # Reset counter if we see sheep again
                no_sheep_count = 0

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, cleaning up...")
        if recorder.is_recording():
            recorder.stop()
            
            # Run analysis on the final video if it exists
            if current_recording_path and os.path.exists(current_recording_path):
                sheep_count = analyze_video(current_recording_path)
                print(f"[FINAL RESULT] Detected {sheep_count} unique sheep in the last video.")
                
        cap.release()

if __name__ == "__main__":
    main()
