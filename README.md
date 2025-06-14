# Livestock Detection and Counting System

This project is designed to detect and count livestock (specifically sheep) in a video stream. It automatically records video segments when livestock are detected and then processes these segments to count the unique animals present.

## Core Functionality

The system operates as follows:

1.  **Video Stream Monitoring:** It continuously monitors a given video stream (e.g., an RTMP stream from a camera).
2.  **Livestock Detection:** Utilizes a YOLO (You Only Look Once) object detection model to identify sheep in the video frames.
3.  **Automatic Recording:** When sheep are detected, the system automatically starts recording the video stream to a local file. Recording continues as long as sheep are present and for a minimum duration to capture relevant footage.
4.  **Video Clip Analysis:** Once a recording is complete, the system processes the video clip to count the number of unique sheep observed. This involves tracking individual animals across frames.
5.  **Output Generation:** The system saves both the raw recorded video clips and the analyzed versions (often with visual markers for detected animals and counts).

## Project Structure

The project consists of the following files:

*   `main.py`: The main executable script. It initializes the system, monitors the video stream, manages the recording state (idle/recording) based on sheep detection, and triggers video analysis.
*   `detection.py`: Contains the `DetectionModel` class, which loads the YOLO model and performs sheep detection on individual video frames.
*   `counting.py`: Implements the `Counting_LiveStocks` class. This class is responsible for processing recorded video clips to identify and count unique sheep, often by tracking them across frames and annotating the video.
*   `recorder.py`: Provides a `Recorder` class that handles the actual video recording process using `ffmpeg` to capture the stream data into a file.
*   `drawing_bounds.py`: Includes functions (`detecting_area`, `draw_bounds`) to define and visualize a specific Region of Interest (ROI) within the video frame where detection and counting should occur.
*   `requirements.txt`: A text file listing all the Python dependencies required to run the project.
*   `yolov8n-seg.pt`: The pre-trained YOLOv8 Nano segmentation model file used for object detection. This is a lightweight model suitable for real-time applications.

## Dependencies

All necessary Python libraries are listed in the `requirements.txt` file.

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the libraries specified in `requirements.txt`, including PyTorch (with CUDA support if available, as per the `--extra-index-url`), Ultralytics YOLO, OpenCV, and others.

## Configuration

Several parameters in `main.py` can be configured to suit your specific setup and needs:

*   `STREAM_URL`: (String) The URL of the video stream to monitor. This is typically an RTMP, RTSP, or HTTP stream URL.
    *   Example: `"rtmp://192.168.180.237/live/streamkey"`
*   `MODEL_NAME`: (String) The filename of the YOLO model to be used for detection. The project currently uses `"yolov8n-seg.pt"`.
*   `POLL_INTERVAL`: (Integer) The time interval in seconds between frame grabs from the stream for detection purposes.
    *   Default: `2` seconds
*   `OUTPUT_DIR`: (String) The directory where raw video clips will be saved when sheep are detected.
    *   Default: `"./raw_clips"`
*   `ANALYSIS_DIR`: (String) The directory where analyzed video clips (with sheep counts and visualizations) will be saved.
    *   Default: `"./analyzed_clips"`
*   `NO_SHEEP_THRESHOLD`: (Integer) The number of consecutive `POLL_INTERVAL`s without detecting sheep before stopping an ongoing recording. This helps prevent stopping the recording due to brief detection misses.
    *   Default: `5` (i.e., if `POLL_INTERVAL` is 2s, recording stops after 10s of no sheep)
*   `MIN_RECORDING_TIME`: (Integer) The minimum duration in seconds that a recording must last, even if sheep are no longer detected. This ensures that very short clips are not created.
    *   Default: `30` seconds

To change these parameters, open `main.py` in a text editor and modify the values assigned to these variables at the beginning of the script.

## Usage Instructions

1.  **Ensure all dependencies are installed** (see [Installation](#installation)).
2.  **Activate your virtual environment** (if you created one):
    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Configure the parameters** in `main.py` as needed, especially `STREAM_URL`.
4.  **Make sure the `yolov8n-seg.pt` model file** is in the same directory as the scripts, or update `MODEL_NAME` in `main.py` if it's located elsewhere.
5.  **Run the main script:**
    ```bash
    python main.py
    ```
6.  The system will start monitoring the stream. Console output will indicate its status (e.g., "Processing frame...", "Sheep detected, starting recording...", "No sheep for Xs, stopping recording.").
7.  To stop the system, press `Ctrl+C` in the terminal. This will attempt to gracefully stop any ongoing recording and perform a final analysis if applicable.

## Output

The system generates the following outputs:

*   **Raw Video Clips:**
    *   Stored in the directory specified by `OUTPUT_DIR` (default: `./raw_clips`).
    *   These are the original video segments recorded directly from the stream when sheep were detected.
    *   Filenames typically include a timestamp, e.g., `clip_YYYYMMDD_HHMMSS.mp4`.
*   **Analyzed Video Clips:**
    *   Stored in the directory specified by `ANALYSIS_DIR` (default: `./analyzed_clips`).
    *   These are the processed versions of the raw clips. The `counting.py` script analyzes these videos to count unique sheep.
    *   The analyzed videos usually have:
        *   Visualizations: Bounding boxes or segmentation masks around detected sheep.
        *   Unique IDs: Tracked sheep may be assigned unique IDs/colors.
        *   Counts: On-screen display of the current number of sheep in sight and the total unique sheep counted in that clip.
    *   Filenames are typically prefixed with `analyzed_`, e.g., `analyzed_clip_YYYYMMDD_HHMMSS.mp4`.
*   **Console Logs:**
    *   The script prints status messages, detection results, recording events, and final sheep counts to the console during operation.

## Model Information

*   **Model Used:** `yolov8n-seg.pt`
*   **Type:** YOLOv8 Nano (segmentation version). This is a lightweight and fast object detection model from the Ultralytics YOLO series, optimized for edge devices and real-time applications.
*   **Role:** The model is responsible for identifying and locating sheep within the video frames. The segmentation capability allows for more precise outlining of the detected animals.
*   **Source:** YOLO models are typically trained on large datasets like COCO. This specific model might be pre-trained or fine-tuned for sheep detection. Ensure you have the correct `.pt` file in your project directory.
