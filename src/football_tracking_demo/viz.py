"""Visualization utilities for drawing tracked players on video frames."""

from collections import defaultdict
from typing import Any

import cv2
import numpy as np


def _id_to_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a given track ID."""
    # Use the golden-ratio hash to spread hues evenly
    hue = int((track_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class TrackVisualizer:
    """Draws bounding boxes, track IDs, and optional motion trails."""

    def __init__(
        self,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_trails: bool = True,
        trail_length: int = 10,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
    ):
        self.show_boxes = show_boxes
        self.show_ids = show_ids
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

        # track_id -> list of (cx, cy) center points for trail drawing
        self._trail_history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[list[float]],
    ) -> np.ndarray:
        """Annotate a frame with tracked detections.

        Args:
            frame: BGR image (H, W, 3). Modified in-place and returned.
            tracks: List of [x1, y1, x2, y2, track_id, confidence].

        Returns:
            The annotated frame (same object as input).
        """
        active_ids = set()

        for trk in tracks:
            x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
            tid = int(trk[4])
            conf = trk[5]
            color = _id_to_color(tid)
            active_ids.add(tid)

            # Center point for trail
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self._trail_history[tid].append((cx, cy))
            if len(self._trail_history[tid]) > self.trail_length:
                self._trail_history[tid] = self._trail_history[tid][
                    -self.trail_length :
                ]

            if self.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            if self.show_ids:
                label = f"ID {tid}  {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.font_thickness,
                )
                # Background rectangle for readability
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (0, 0, 0),
                    self.font_thickness,
                )

            if self.show_trails:
                pts = self._trail_history[tid]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)  # fade older points
                    thickness = max(1, int(self.box_thickness * alpha))
                    cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        # Prune stale trails for IDs no longer active
        stale = [tid for tid in self._trail_history if tid not in active_ids]
        for tid in stale:
            del self._trail_history[tid]

        return frame

    def reset(self) -> None:
        """Clear trail history."""
        self._trail_history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "TrackVisualizer":
        """Create from the visualization section of config.yaml."""
        cfg = config or {}
        return cls(
            show_boxes=cfg.get("show_boxes", True),
            show_ids=cfg.get("show_ids", True),
            show_trails=cfg.get("show_trails", True),
            trail_length=cfg.get("trail_length", 10),
            box_thickness=cfg.get("box_thickness", 2),
            font_scale=cfg.get("font_scale", 0.6),
            font_thickness=cfg.get("font_thickness", 2),
        )
