"""Player detection using YOLOv8 with HUD masking."""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from football_tracking_demo.filtering import filter_detections


def apply_hud_mask(
    frame: np.ndarray,
    top_percent: float = 0.10,
    bottom_percent: float = 0.12,
) -> np.ndarray:
    """Black out HUD/overlay regions of a broadcast frame."""
    masked = frame.copy()
    h = masked.shape[0]
    masked[: int(h * top_percent), :] = 0
    masked[int(h * (1.0 - bottom_percent)) :, :] = 0
    return masked


class PlayerDetector:
    """YOLO-x-based person detector with HUD masking."""

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        model_dir: str = "checkpoints",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        hud_top: float = 0.10,
        hud_bottom: float = 0.12,
        hud_enabled: bool = True,
        shape_filter_config: dict[str, Any] | None = None,
        field_mask_config: dict[str, Any] | None = None,
    ):
        model_path = Path(model_dir) / model_name
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Load from cache or download and cache
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO(model_name)
            default_weight = Path(model_name)
            if default_weight.exists() and not model_path.exists():
                default_weight.rename(model_path)
                self.model = YOLO(str(model_path))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.hud_top = hud_top
        self.hud_bottom = hud_bottom
        self.hud_enabled = hud_enabled
        self.shape_filter_config = shape_filter_config
        self.field_mask_config = field_mask_config

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Detect persons in a single frame.

        Returns:
            List of [x1, y1, x2, y2, confidence] per detection.
        """
        if self.hud_enabled:
            input_frame = apply_hud_mask(frame, self.hud_top, self.hud_bottom)
        else:
            input_frame = frame

        results = self.model.predict(
            input_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: list[list[float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    [
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                        conf,
                    ]
                )

        return detections

    def detect_and_filter(self, frame: np.ndarray) -> list[list[float]]:
        """Detect persons and apply playing field filtering in one step."""
        detections = self.detect(frame)
        return filter_detections(
            detections, frame, self.shape_filter_config, self.field_mask_config
        )
