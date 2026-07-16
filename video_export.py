"""Browser-compatible MP4 export helpers."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def add_canvas_padding(frame, padding=24, color=(255, 255, 255)):
    """Add an even outer margin without scaling or cropping the content."""
    if padding < 0:
        raise ValueError("video padding cannot be negative")
    if padding == 0:
        return frame
    height, width = frame.shape[:2]
    canvas = np.full(
        (height + 2 * padding, width + 2 * padding, 3),
        color,
        dtype=np.uint8,
    )
    canvas[padding : padding + height, padding : padding + width] = frame
    return canvas


class H264VideoWriter:
    """Write frames with OpenCV, then atomically publish an H.264 MP4.

    OpenCV's broadly available ``mp4v`` encoder produces MPEG-4 Part 2, which
    Chromium-based previews such as VS Code often cannot decode.  A temporary
    OpenCV stream is therefore transcoded with FFmpeg to H.264/yuv420p.
    """

    def __init__(self, path, fps, frame_size, crf=18):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for H.264 video export but was not found in PATH"
            )
        width, height = frame_size
        if width % 2 or height % 2:
            raise ValueError(
                f"H.264 yuv420p requires even dimensions, got {width}x{height}"
            )
        self.fps = fps
        self.crf = crf
        self._closed = False
        self._raw_path = self._temporary_path(".raw.mp4")
        self._encoded_path = self._temporary_path(".h264.mp4")
        self._writer = cv2.VideoWriter(
            str(self._raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not self._writer.isOpened():
            self._cleanup()
            raise RuntimeError(f"cannot open temporary video writer for {self.path}")

    def _temporary_path(self, suffix):
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{self.path.stem}.",
            suffix=suffix,
            dir=self.path.parent,
            delete=False,
        )
        path = Path(handle.name)
        handle.close()
        return path

    def write(self, frame):
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")
        self._writer.write(frame)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._writer.release()
        command = [
            self.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(self._raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._encoded_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = result.stderr.strip() or "unknown FFmpeg error"
                raise RuntimeError(f"H.264 video encoding failed: {detail}")
            os.replace(self._encoded_path, self.path)
        finally:
            self._cleanup()

    def abort(self):
        if not self._closed:
            self._closed = True
            self._writer.release()
        self._cleanup()

    def _cleanup(self):
        for path in (self._raw_path, self._encoded_path):
            path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
