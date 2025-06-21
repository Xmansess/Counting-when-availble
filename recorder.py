import subprocess
import shlex
import logging
import threading
import time

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.process = None
        self.output_path = None
        self._stop_event = threading.Event()
        self._monitor_thread = None
        logger.info(f"Recorder initialized for stream URL: {self.stream_url}")

    def _monitor_ffmpeg_process(self):
        """Monitors the ffmpeg process for unexpected termination and logs stderr."""
        logger.debug(f"FFmpeg monitor thread started for {self.output_path}.")

        # Read stderr line by line
        try:
            if self.process and self.process.stderr:
                for line in iter(self.process.stderr.readline, b''):
                    if self._stop_event.is_set():
                        logger.debug("FFmpeg monitor: Stop event received, exiting stderr loop.")
                        break
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str: # Log only non-empty lines
                        logger.debug(f"FFmpeg stderr ({self.output_path}): {line_str}")
            else:
                logger.warning("FFmpeg process or stderr not available at start of monitor thread.")

        except Exception as e:
            logger.error(f"Exception in FFmpeg stderr monitoring loop for {self.output_path}: {e}", exc_info=True)
        finally:
            # Wait for the process to complete and get return code
            if self.process:
                self.process.wait() # Ensure process is waited on
                return_code = self.process.returncode
                if return_code == 0 or return_code is None: # None if terminate() was called and already handled
                     logger.info(f"FFmpeg process for {self.output_path} finished with return code {return_code}.")
                elif return_code == -15: # SIGTERM, expected on self.stop()
                    logger.info(f"FFmpeg process for {self.output_path} terminated as expected (SIGTERM).")
                else:
                    logger.warning(f"FFmpeg process for {self.output_path} exited unexpectedly with return code: {return_code}")
            logger.debug(f"FFmpeg monitor thread finished for {self.output_path}.")


    def start(self, output_path: str):
        if self.process is not None:
            logger.warning(f"Recording already in progress to {self.output_path}. Start command for {output_path} ignored.")
            return

        self.output_path = output_path
        self._stop_event.clear() # Clear stop event for the new recording session

        # Using -nostdin to prevent ffmpeg from consuming stdin, which can be an issue in some environments
        cmd = f"ffmpeg -y -nostdin -i {self.stream_url} -c copy {shlex.quote(self.output_path)}"

        logger.info(f"Starting recording to: {self.output_path}")
        logger.debug(f"Executing FFmpeg command: {cmd}")

        try:
            # Start the ffmpeg process
            self.process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE, # Capture stdout (though -c copy might not produce much)
                stderr=subprocess.PIPE  # Capture stderr for logging
            )
            logger.info(f"FFmpeg process started for {self.output_path} with PID {self.process.pid}.")

            # Start the monitoring thread
            self._monitor_thread = threading.Thread(target=self._monitor_ffmpeg_process, daemon=True)
            self._monitor_thread.start()

        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            self.process = None # Ensure process is None if Popen fails
        except Exception as e:
            logger.error(f"Failed to start ffmpeg process for {self.output_path}: {e}", exc_info=True)
            self.process = None # Ensure process is None if Popen fails


    def stop(self):
        if not self.process:
            logger.info("Stop called but no recording process active.")
            return

        logger.info(f"Stopping recording for {self.output_path} (PID: {self.process.pid}).")
        self._stop_event.set() # Signal the monitor thread to stop reading stderr

        try:
            # Check if process is still running
            if self.process.poll() is None: # Process is still running
                logger.debug(f"Terminating FFmpeg process {self.process.pid}...")
                self.process.terminate() # Send SIGTERM

                # Wait for a short period for graceful termination
                try:
                    self.process.wait(timeout=5) # Wait up to 5 seconds
                    logger.info(f"FFmpeg process {self.process.pid} terminated gracefully with code {self.process.returncode}.")
                except subprocess.TimeoutExpired:
                    logger.warning(f"FFmpeg process {self.process.pid} did not terminate in time. Sending SIGKILL.")
                    self.process.kill() # Force kill if terminate didn't work
                    self.process.wait() # Wait for kill to complete
                    logger.info(f"FFmpeg process {self.process.pid} killed.")
            else:
                logger.info(f"FFmpeg process {self.process.pid} already exited with code {self.process.returncode} before explicit stop.")

        except Exception as e:
            logger.error(f"Exception during FFmpeg stop for {self.output_path}: {e}", exc_info=True)

        finally:
            if self._monitor_thread and self._monitor_thread.is_alive():
                logger.debug(f"Waiting for FFmpeg monitor thread to join for {self.output_path}...")
                self._monitor_thread.join(timeout=2.0) # Wait for monitor thread
                if self._monitor_thread.is_alive():
                    logger.warning(f"FFmpeg monitor thread for {self.output_path} did not join in time.")

            self.process = None
            self.output_path = None # Clear output path after stopping
            logger.info("Recording process stopped and resources cleaned up.")


    def is_recording(self) -> bool:
        # Check if the process object exists and if it's still running
        is_rec = self.process is not None and self.process.poll() is None
        logger.debug(f"is_recording check: {is_rec}")
        return is_rec
