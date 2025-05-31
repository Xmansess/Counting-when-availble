# main.py
import os
import time
import cv2 as cv
from datetime import datetime
import subprocess
import threading # <--- Ավելացրել ենք threading
from queue import Queue, Empty # <--- Ավելացրել ենք Queue

from detection import DetectionModel
from recorder import Recorder
from drawing_bounds import detecting_area
from counting import Counting_LiveStocks

# === Configuration ===
STREAM_URL = "rtmp://localhost/live/stream"
MODEL_NAME = "yolov8n-seg.pt"
POLL_INTERVAL = 2  # seconds (How often to process a frame)
OUTPUT_DIR = "./raw_clips"
ANALYSIS_DIR = "./analyzed_clips"

# Parameters to make the system more stable
NO_SHEEP_THRESHOLD = 5  # How many intervals to wait before stopping recording
MIN_RECORDING_TIME = 30  # Minimum recording time in seconds

# --- Threading Components ---
# Queue to hold the latest frame
# maxsize=1 ensures we only keep the most recent frame, discarding older ones if the reader is faster than the consumer
frame_queue = Queue(maxsize=1)
# Event to signal the reader thread to stop
stop_event = threading.Event()
# --- End Threading Components ---


# Ensure output folders exist
def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Frame Reader Thread ---
# This function runs in a separate thread and continuously reads frames
def frame_reader_thread(url, queue, stop_signal):
    print("[INFO] Frame reader thread started.")
    cap = None
    while not stop_signal.is_set():
        if cap is None or not cap.isOpened():
            print("[INFO] Reader: Opening video capture...")
            cap = cv.VideoCapture(url)
            if not cap.isOpened():
                print("[ERROR] Reader: Failed to open stream. Retrying in 5s...")
                time.sleep(5)
                continue
            else:
                 # Attempt to reduce buffer size - might help but not guaranteed
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                print("[INFO] Reader: Stream opened successfully.")

        ret, frame = cap.read()

        if not ret:
            print("[WARN] Reader: Failed to grab frame. Re-opening stream...")
            cap.release()
            cap = None # Signal to reopen in the next loop iteration
            time.sleep(1) # Brief pause before retry
            continue

        # If the queue is full, remove the old frame before adding the new one
        # This ensures the queue always holds the latest frame
        if queue.full():
            try:
                queue.get_nowait() # Remove the old frame
            except Empty:
                pass # Should not happen if full, but safety check

        try:
            queue.put_nowait(frame) # Put the latest frame
        except Full:
             pass # Should not happen with maxsize=1 and the check above

        # Optional: Slight sleep if needed, e.g., if read() returns too fast
        # time.sleep(0.01) # Can help reduce CPU if read is non-blocking and very fast

    # Cleanup when stop signal is received
    if cap is not None and cap.isOpened():
        cap.release()
    print("[INFO] Frame reader thread stopped.")
# --- End Frame Reader Thread ---


# Run the counting analysis on a completed video
def analyze_video(video_path):
    print(f"[INFO] Analyzing video: {video_path}")
    # Ensure the analysis function uses the correct model path if needed
    counter = Counting_LiveStocks(MODEL_NAME, video_path, ANALYSIS_DIR)
    counter() # Assuming this runs the analysis and saves results
    # Modify this based on how Counting_LiveStocks returns the count
    sheep_count = getattr(counter, 'final_sheep_count', len(getattr(counter, 'id_color', {}).keys())) # Try to get a final count attribute or fallback
    print(f"[INFO] Analysis complete. Counted {sheep_count} unique sheep.")
    return sheep_count

# === Main state machine ===
def main():
    print("[INFO] Starting main loop...")
    setup_dirs()
    detector = DetectionModel(MODEL_NAME)
    # Recorder might need adjustment if it reads the stream independently.
    # Assuming it uses ffmpeg subprocess based on URL, which is fine.
    recorder = Recorder(STREAM_URL)

    # Start the frame reader thread
    reader_thread = threading.Thread(target=frame_reader_thread, args=(STREAM_URL, frame_queue, stop_event), daemon=True)
    reader_thread.start()

    state = "idle"

    # Stabilization variables
    no_sheep_count = 0
    recording_start_time = 0
    current_recording_path = None
    last_processed_time = time.time() # Track when we last processed a frame

    try:
        while True:
            # Check if the reader thread is alive
            if not reader_thread.is_alive() and not stop_event.is_set():
                 print("[ERROR] Frame reader thread has unexpectedly stopped. Exiting.")
                 # Optionally try to restart the thread, or exit
                 break # Exit for now

            # --- Get the latest frame from the queue ---
            current_frame = None
            try:
                # Get the most recent frame without waiting
                current_frame = frame_queue.get_nowait()
                # print("[DEBUG] Got frame from queue") # Optional debug
            except Empty:
                # No new frame available since last check, wait before retrying
                # print("[DEBUG] Frame queue empty") # Optional debug
                # Check if enough time has passed to process again anyway
                if time.time() - last_processed_time < POLL_INTERVAL:
                    time.sleep(0.1) # Short sleep to avoid busy-waiting
                    continue # Skip processing if queue is empty and not enough time passed
                else:
                    # If queue is empty but POLL_INTERVAL passed, maybe process last known state?
                    # Or just log a warning and wait longer. For now, just proceed carefully.
                    print("[WARN] Frame queue empty, but processing interval elapsed. Potential stream issue?")
                    # We don't have a frame, so we act as if no sheep were detected
                    has_sheep = False
                    # Fall through to state machine logic with has_sheep = False

            # --- Process the frame (if we got one) ---
            if current_frame is not None:
                print("[INFO] Processing frame...")  # Debug info
                # Assuming detecting_area is fast or part of detection pipeline
                frame_proc = detecting_area(current_frame)
                # Use the detector instance
                has_sheep = detector.detect(frame_proc, conf_threshold=0.5) # Ensure detector.detect returns boolean
                print(f"[INFO] Sheep detected: {has_sheep}")  # Debug info
                last_processed_time = time.time() # Update time only when a frame is processed
            # else: # If current_frame is None (due to queue empty timeout)
                # has_sheep is already set to False above


            # === State machine logic (mostly unchanged) ===
            if has_sheep and state == "idle":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_recording_path = os.path.join(OUTPUT_DIR, f"clip_{ts}.mp4")
                print(f"[INFO] Sheep detected, starting recording to {current_recording_path}")
                # Make sure recorder.start doesn't block for too long
                recorder.start(current_recording_path)
                recording_start_time = time.time()
                state = "recording"
                no_sheep_count = 0 # Reset counter

            elif not has_sheep and state == "recording":
                no_sheep_count += 1
                current_duration = time.time() - recording_start_time

                # Use calculated time without sheep instead of count * interval for more accuracy
                time_since_last_sheep_seen = no_sheep_count * POLL_INTERVAL # Approximate time

                print(f"[DEBUG] No sheep detected. Count: {no_sheep_count}, Time since last sheep approx: {time_since_last_sheep_seen}s") # Debug

                # Stop condition uses the count threshold and minimum duration
                if no_sheep_count >= NO_SHEEP_THRESHOLD and current_duration >= MIN_RECORDING_TIME:
                    print(f"[INFO] No sheep confirmed for ~{time_since_last_sheep_seen}s (threshold: {NO_SHEEP_THRESHOLD * POLL_INTERVAL}s) and duration > {MIN_RECORDING_TIME}s. Stopping recording.")
                    recorder.stop()
                    state = "idle"

                    # Run analysis on the completed video
                    if current_recording_path and os.path.exists(current_recording_path):
                        # Consider running analysis in a separate thread/process if it's slow
                        sheep_count = analyze_video(current_recording_path)
                        print(f"[RESULT] Detected {sheep_count} unique sheep in the video.")
                        current_recording_path = None # Reset path after analysis
                    else:
                        print("[WARN] Recording file not found or path invalid for analysis.")

                elif current_duration < MIN_RECORDING_TIME:
                     print(f"[INFO] No sheep detected, but recording duration ({current_duration:.1f}s) is less than minimum ({MIN_RECORDING_TIME}s). Continuing recording.")
                     # Still increment no_sheep_count, but don't stop yet
                else:
                    # Duration is sufficient, but no_sheep_count hasn't reached threshold
                    print(f"[INFO] No sheep detected for ~{time_since_last_sheep_seen}s, continuing recording (Threshold: {NO_SHEEP_THRESHOLD * POLL_INTERVAL}s).")


            elif has_sheep and state == "recording":
                # Reset counter if we see sheep again while recording
                if no_sheep_count > 0:
                    print("[INFO] Sheep re-detected, resetting no-sheep counter.")
                    no_sheep_count = 0

            elif not has_sheep and state == "idle":
                # Optional: Add a print statement if needed
                # print("[INFO] Idle and no sheep detected.")
                pass # Do nothing


            # --- Wait before processing the next frame ---
            # We already got the latest frame, so the main loop waits POLL_INTERVAL
            # This controls how frequently we RUN the detection, not how frequently we read
            time.sleep(POLL_INTERVAL)


    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, cleaning up...")
        # Signal the reader thread to stop and wait for it
        stop_event.set()
        if reader_thread.is_alive():
             print("[INFO] Waiting for frame reader thread to stop...")
             reader_thread.join(timeout=5.0) # Wait for thread with timeout
             if reader_thread.is_alive():
                  print("[WARN] Frame reader thread did not stop gracefully.")

        # Stop recorder if it's running
        if recorder.is_recording():
            print("[INFO] Stopping active recording...")
            recorder.stop()

            # Run analysis on the final video if it exists and recording was active
            if current_recording_path and os.path.exists(current_recording_path):
                print("[INFO] Analyzing final recorded clip...")
                sheep_count = analyze_video(current_recording_path)
                print(f"[FINAL RESULT] Detected {sheep_count} unique sheep in the last video.")
            else:
                 print("[WARN] Final recording file not found or path invalid.")

    finally:
        # Ensure thread is signaled to stop even on other exceptions
        if not stop_event.is_set():
            stop_event.set()
            if reader_thread.is_alive():
                 reader_thread.join(timeout=2.0)

        print("[INFO] Main loop finished.")


if __name__ == "__main__":
    main()
