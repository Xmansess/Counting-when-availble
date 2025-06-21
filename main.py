# main.py
import os
import time
import cv2 as cv
from datetime import datetime
import subprocess
import threading
from queue import Queue, Empty, Full # Added Full for explicit exception handling
import logging # Added logging

from detection import DetectionModel
from recorder import Recorder
from drawing_bounds import detecting_area
from counting import Counting_LiveStocks
from logger_config import setup_logger # Import the setup function

# Initialize logger
logger = setup_logger()

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
frame_queue = Queue(maxsize=1)
stop_event = threading.Event()
# --- End Threading Components ---


# Ensure output folders exist
def setup_dirs():
    logger.info(f"Ensuring output directory exists: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensuring analysis directory exists: {ANALYSIS_DIR}")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Frame Reader Thread ---
def frame_reader_thread(url, queue, stop_signal):
    logger.info("Frame reader thread started.")
    cap = None
    while not stop_signal.is_set():
        if cap is None or not cap.isOpened():
            logger.info("Reader: Opening video capture...")
            cap = cv.VideoCapture(url)
            if not cap.isOpened():
                logger.error(f"Reader: Failed to open stream: {url}. Retrying in 5s...")
                time.sleep(5)
                continue
            else:
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1) # Attempt to reduce buffer size
                logger.info(f"Reader: Stream opened successfully: {url}")

        ret, frame = cap.read()

        if not ret:
            logger.warning("Reader: Failed to grab frame. Re-opening stream...")
            if cap is not None:
                cap.release()
            cap = None
            time.sleep(1)
            continue

        if queue.full():
            try:
                queue.get_nowait()
                logger.debug("Reader: Removed old frame from full queue.")
            except Empty:
                logger.warning("Reader: Queue was full but get_nowait failed. This shouldn't happen.")
                pass

        try:
            queue.put_nowait(frame)
            logger.debug("Reader: Put new frame into queue.")
        except Full:
            logger.warning("Reader: Queue full on put_nowait, even after checking. Frame dropped.")
            pass

    if cap is not None and cap.isOpened():
        cap.release()
    logger.info("Frame reader thread stopped.")
# --- End Frame Reader Thread ---


# Run the counting analysis on a completed video
def analyze_video(video_path):
    logger.info(f"Analyzing video: {video_path}")
    try:
        counter = Counting_LiveStocks(MODEL_NAME, video_path, ANALYSIS_DIR)
        counter()
        sheep_count = getattr(counter, 'final_sheep_count', len(getattr(counter, 'id_color', {}).keys()))
        logger.info(f"Analysis complete for {video_path}. Counted {sheep_count} unique sheep.")
        return sheep_count
    except Exception as e:
        logger.error(f"Error during video analysis for {video_path}: {e}", exc_info=True)
        return 0 # Return a default value or re-raise

# === Main state machine ===
def main():
    logger.info("Starting Livestock Detection and Counting System.")
    logger.info("--- Configuration ---")
    logger.info(f"STREAM_URL: {STREAM_URL}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"POLL_INTERVAL: {POLL_INTERVAL}s")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"ANALYSIS_DIR: {ANALYSIS_DIR}")
    logger.info(f"NO_SHEEP_THRESHOLD: {NO_SHEEP_THRESHOLD} intervals")
    logger.info(f"MIN_RECORDING_TIME: {MIN_RECORDING_TIME}s")
    logger.info("---------------------")

    setup_dirs()

    try:
        detector = DetectionModel(MODEL_NAME)
        recorder = Recorder(STREAM_URL)
    except Exception as e:
        logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
        return # Cannot proceed

    reader_thread = threading.Thread(target=frame_reader_thread, args=(STREAM_URL, frame_queue, stop_event), daemon=True)
    logger.info("Starting frame reader thread.")
    reader_thread.start()

    state = "idle"
    logger.info(f"Initial state: {state}")

    no_sheep_count = 0
    recording_start_time = 0
    current_recording_path = None
    last_processed_time = time.time()
    processing_complete = False # Flag to indicate when to stop the main loop

    try:
        while not processing_complete: # Loop until processing is marked complete
            if not reader_thread.is_alive() and not stop_event.is_set():
                 logger.error("Frame reader thread has unexpectedly stopped. Exiting.")
                 processing_complete = True # Stop main loop if reader dies
                 continue


            current_frame = None
            try:
                current_frame = frame_queue.get_nowait()
                logger.debug(f"Got frame from queue. Queue size approx: {frame_queue.qsize()}")
            except Empty:
                logger.debug("Frame queue empty.")
                if time.time() - last_processed_time < POLL_INTERVAL:
                    time.sleep(0.1)
                    continue
                else:
                    logger.warning("Frame queue empty, but processing interval elapsed. Potential stream issue? Assuming no sheep.")
                    has_sheep = False

            if current_frame is not None:
                logger.info("Processing frame for sheep detection...")
                frame_proc = detecting_area(current_frame) # Assuming this is lightweight
                has_sheep = detector.detect(frame_proc, conf_threshold=0.5)
                logger.info(f"Sheep detected: {has_sheep}")
                last_processed_time = time.time()
            # If current_frame was None due to queue empty timeout, has_sheep is already False

            if state == "idle":
                if has_sheep:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_recording_path = os.path.join(OUTPUT_DIR, f"clip_{ts}.mp4")
                    logger.info(f"State change: idle -> recording. Sheep detected, starting recording to {current_recording_path}")
                    recorder.start(current_recording_path)
                    recording_start_time = time.time()
                    state = "recording"
                    no_sheep_count = 0
                else:
                    logger.debug(f"State: idle. No sheep detected.")

            elif state == "recording":
                if has_sheep:
                    if no_sheep_count > 0:
                        logger.info("Sheep re-detected during recording, resetting no-sheep counter.")
                        no_sheep_count = 0
                else: # No sheep while recording
                    no_sheep_count += 1
                    current_duration = time.time() - recording_start_time
                    time_since_last_sheep_seen = no_sheep_count * POLL_INTERVAL

                    logger.debug(f"State: recording. No sheep detected. No-sheep count: {no_sheep_count}, Time since last sheep approx: {time_since_last_sheep_seen}s")

                    if no_sheep_count >= NO_SHEEP_THRESHOLD and current_duration >= MIN_RECORDING_TIME:
                        logger.info(f"State change: recording -> idle. No sheep confirmed for ~{time_since_last_sheep_seen}s (threshold: {NO_SHEEP_THRESHOLD * POLL_INTERVAL}s) and duration ({current_duration:.1f}s) > min_recording_time ({MIN_RECORDING_TIME}s). Stopping recording.")
                        recorder.stop()
                        state = "idle"

                        if current_recording_path and os.path.exists(current_recording_path):
                            logger.info(f"Initiating analysis for {current_recording_path}")
                            # Consider running analysis in a separate thread/process if it's very slow
                            # For now, keeping it synchronous as per original code structure.
                            analyze_video(current_recording_path) # Result already logged in analyze_video
                            current_recording_path = None
                            logger.info("Video analysis complete. Setting processing_complete to True.")
                            processing_complete = True # Signal to exit the main loop
                        else:
                            logger.warning(f"Recording file not found or path invalid for analysis: {current_recording_path}")
                            # Decide if we should stop anyway if the file is missing
                            # For now, let's assume if the file is gone, something is wrong, and we should stop.
                            logger.info("Setting processing_complete to True due to missing recording file for analysis.")
                            processing_complete = True # Signal to exit the main loop

                    elif current_duration < MIN_RECORDING_TIME:
                         logger.info(f"State: recording. No sheep detected, but recording duration ({current_duration:.1f}s) is less than minimum ({MIN_RECORDING_TIME}s). Continuing recording.")
                    else:
                        logger.info(f"State: recording. No sheep detected for ~{time_since_last_sheep_seen}s, continuing recording (Threshold: {NO_SHEEP_THRESHOLD * POLL_INTERVAL}s reached, but min duration might not be).")

            logger.debug(f"Main loop waiting for {POLL_INTERVAL} seconds before next detection cycle.")
            time.sleep(POLL_INTERVAL)

        if processing_complete:
            logger.info("Primary processing complete. Moving to final cleanup.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Cleaning up and shutting down...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("Initiating shutdown sequence.")
        stop_event.set()
        logger.info("Stop event set for frame reader thread.")

        if reader_thread.is_alive():
             logger.info("Waiting for frame reader thread to stop...")
             reader_thread.join(timeout=5.0)
             if reader_thread.is_alive():
                  logger.warning("Frame reader thread did not stop gracefully.")

        if recorder.is_recording():
            logger.info("Stopping active recording due to shutdown...")
            recorder.stop()
            if current_recording_path and os.path.exists(current_recording_path):
                logger.info(f"Analyzing final recorded clip: {current_recording_path}")
                analyze_video(current_recording_path)
            else:
                 logger.warning(f"Final recording file not found or path invalid during shutdown: {current_recording_path}")

        logger.info("Livestock Detection and Counting System shut down.")


if __name__ == "__main__":
    main()
