"""Video I/O utilities for reading and writing video files."""

from pathlib import Path
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np


def get_video_metadata(path: str) -> dict[str, Any]:
    """
    Extract metadata from a video file.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with keys: fps, width, height, frame_count, duration.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If the video file cannot be opened.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": duration,
        }
    finally:
        cap.release()


def load_video(path: str) -> Generator[np.ndarray, None, None]:
    """
    Load video frames as a generator.

    Args:
        path: Path to the video file.

    Yields:
        BGR numpy arrays (H, W, 3) for each frame.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If the video file cannot be opened.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


class VideoWriter:
    """Wrapper around OpenCV VideoWriter for saving annotated video output."""

    def __init__(
        self, path: str, fps: float, width: int, height: int, codec: str = "mp4v"
    ):
        """
        Initialize video writer.

        Args:
            path: Output video file path.
            fps: Frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FourCC codec string (default: mp4v).
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for: {path}")

        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to the video file."""
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def release(self) -> None:
        """Release the video writer and finalize the file."""
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
