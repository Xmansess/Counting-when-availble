import subprocess
import shlex

class Recorder:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.process = None

    def start(self, output_path: str):
        if self.process is not None:
            return  # already recording
        # ffmpeg copy stream directly to file
        cmd = f"ffmpeg -y -i {self.stream_url} -c copy {output_path}"
        self.process = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def stop(self):
        if not self.process:
            return
        self.process.terminate()
        self.process.wait()
        self.process = None

    def is_recording(self) -> bool:
        return self.process is not None
