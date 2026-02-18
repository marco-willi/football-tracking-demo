"""Multi-object tracking using ByteTrack via the supervision library."""

from typing import Any

import numpy as np
import supervision as sv


class ByteTracker:
    """Wraps supervision.ByteTrack to provide a simple per-frame update API.

    Each call to ``update()`` accepts raw detections and returns tracked
    detections with stable IDs in the format expected by the rest of the
    pipeline: ``[x1, y1, x2, y2, track_id, confidence]``.
    """

    def __init__(
        self,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        track_activation_threshold: float = 0.25,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ):
        """
        Args:
            track_buffer: Number of frames to keep lost tracks alive.
            match_threshold: Minimum IOU for matching detections to tracks.
            track_activation_threshold: Confidence above which a detection
                can initiate a new track.
            frame_rate: Expected video frame rate (used internally by ByteTrack).
            minimum_consecutive_frames: How many consecutive detections before
                a track is considered confirmed.
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: list[list[float]]) -> list[list[float]]:
        """Feed detections for a single frame and return tracked results.

        Args:
            detections: List of ``[x1, y1, x2, y2, confidence]``.

        Returns:
            List of ``[x1, y1, x2, y2, track_id, confidence]``.
            ``track_id`` is a stable integer ID assigned by ByteTrack.
        """
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            arr = np.array(detections, dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=arr[:, :4],
                confidence=arr[:, 4],
            )

        tracked: sv.Detections = self._tracker.update_with_detections(sv_dets)

        results: list[list[float]] = []
        if tracked.tracker_id is None or len(tracked) == 0:
            return results

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tid = int(tracked.tracker_id[i])
            conf = (
                float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            )
            results.append([float(x1), float(y1), float(x2), float(y2), tid, conf])

        return results

    def reset(self) -> None:
        """Reset tracker state (e.g. between video clips)."""
        self._tracker.reset()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "ByteTracker":
        """Create a ByteTracker from the tracking section of config.yaml."""
        cfg = config or {}
        return cls(
            track_buffer=cfg.get("track_buffer", 30),
            match_threshold=cfg.get("match_threshold", 0.8),
            frame_rate=cfg.get("frame_rate", 30),
            track_activation_threshold=cfg.get("track_activation_threshold", 0.25),
        )
