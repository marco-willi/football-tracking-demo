"""Multi-object tracking — ByteTrack and BoT-SORT backends."""

from __future__ import annotations

import warnings
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

    def update(
        self,
        detections: list[list[float]],
        frame: np.ndarray
        | None = None,  # accepted but unused — API parity with BotSortTracker
    ) -> list[list[float]]:
        """Feed detections for a single frame and return tracked results.

        Args:
            detections: List of ``[x1, y1, x2, y2, confidence]``.
            frame: Ignored by ByteTrack (present for API parity with BotSortTracker).

        Returns:
            List of ``[x1, y1, x2, y2, track_id, confidence]``.
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


class BotSortTracker:
    """Wraps boxmot BotSort with camera motion compensation (CMC).

    Unlike ByteTrack, BoT-SORT estimates the inter-frame camera transform
    (sparse optical flow by default) and corrects all Kalman predictions before
    IoU matching. This greatly reduces track fragmentation on broadcast footage
    where the camera pans or tilts continuously.

    Requires: ``pip install boxmot``

    ReID is disabled by default (``reid_weights=None``). Pass a path to an
    osnet/resnet ReID weight file to enable appearance-based re-identification.
    """

    def __init__(
        self,
        device: str = "cpu",
        half: bool = False,
        reid_weights: str | None = None,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        track_activation_threshold: float = 0.25,
        frame_rate: int = 30,
        cmc_method: str = "sparseOptFlow",
    ):
        try:
            import torch
            from pathlib import Path

            from boxmot import BotSort
        except ImportError as e:
            raise ImportError(
                "boxmot is required for BoT-SORT. Install with: pip install boxmot"
            ) from e

        import torch
        from pathlib import Path

        from boxmot import BotSort

        reid_path = Path(reid_weights) if reid_weights else Path("noreid.pt")

        self._tracker = BotSort(
            reid_weights=reid_path,
            device=torch.device(device),
            half=half,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            new_track_thresh=track_activation_threshold,
            frame_rate=frame_rate,
            cmc_method=cmc_method,
            with_reid=reid_weights is not None,
        )

    def update(
        self,
        detections: list[list[float]],
        frame: np.ndarray | None = None,
    ) -> list[list[float]]:
        """Feed detections for a single frame and return tracked results.

        Args:
            detections: List of ``[x1, y1, x2, y2, confidence]``.
            frame: BGR frame (H, W, 3) uint8. Required for CMC — if omitted a
                black dummy frame is used and camera motion is not compensated.

        Returns:
            List of ``[x1, y1, x2, y2, track_id, confidence]``.
        """
        if not detections:
            return []

        arr = np.array(detections, dtype=np.float32)
        # boxmot format: [x1, y1, x2, y2, conf, class_id]
        dets = np.hstack([arr[:, :5], np.zeros((len(arr), 1), dtype=np.float32)])

        if frame is None:
            warnings.warn(
                "BotSortTracker.update() called without a frame — "
                "CMC is disabled and camera motion will not be compensated.",
                stacklevel=2,
            )
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tracks = self._tracker.update(dets, frame)

        results: list[list[float]] = []
        if tracks is not None and len(tracks) > 0:
            for row in tracks:
                results.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        int(row[4]),
                        float(row[5]),
                    ]
                )

        return results

    def reset(self) -> None:
        """Reset tracker state (e.g. between video clips)."""
        self._tracker.reset()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "BotSortTracker":
        """Create a BotSortTracker from the tracking section of config.yaml."""
        cfg = config or {}
        bot = cfg.get("botsort", {})
        return cls(
            device=bot.get("device", "cpu"),
            half=bot.get("half", False),
            reid_weights=bot.get("reid_weights"),  # None = no ReID
            track_buffer=cfg.get("track_buffer", 30),
            match_threshold=cfg.get("match_threshold", 0.8),
            track_activation_threshold=cfg.get("track_activation_threshold", 0.25),
            frame_rate=cfg.get("frame_rate", 30),
            cmc_method=bot.get("cmc_method", "sparseOptFlow"),
        )


def build_tracker(
    config: dict[str, Any] | None = None,
) -> ByteTracker | BotSortTracker:
    """Instantiate the tracker named in ``config['tracker']``.

    Supported values (case-insensitive):
        - ``"bytetrack"`` (default) — supervision ByteTrack
        - ``"botsort"``             — boxmot BoT-SORT with CMC
    """
    cfg = config or {}
    name = cfg.get("tracker", "bytetrack").lower().replace("-", "").replace("_", "")
    if name == "botsort":
        return BotSortTracker.from_config(cfg)
    return ByteTracker.from_config(cfg)
