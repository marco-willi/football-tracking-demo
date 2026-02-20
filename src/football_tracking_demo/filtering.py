"""Playing field filtering to remove false-positive detections."""

from typing import Any

import cv2
import numpy as np


def _bbox_size(det: list[float]) -> tuple[float, float]:
    """Return (width, height) of a detection [x1, y1, x2, y2, conf]."""
    return det[2] - det[0], det[3] - det[1]


def filter_by_size(
    detections: list[list[float]],
    min_w: float = 20,
    min_h: float = 40,
    max_w: float = 300,
    max_h: float = 500,
) -> list[list[float]]:
    """Remove detections whose bounding box is too small or too large."""
    kept = []
    for det in detections:
        w, h = _bbox_size(det)
        if min_w <= w <= max_w and min_h <= h <= max_h:
            kept.append(det)
    return kept


def filter_by_aspect_ratio(
    detections: list[list[float]],
    min_ratio: float = 0.3,
    max_ratio: float = 4.0,
) -> list[list[float]]:
    """Remove detections with extreme height/width aspect ratio."""
    kept = []
    for det in detections:
        w, h = _bbox_size(det)
        if w == 0:
            continue
        ratio = h / w
        if min_ratio <= ratio <= max_ratio:
            kept.append(det)
    return kept


def build_playing_field_mask(
    frame: np.ndarray,
    hsv_lower: tuple[int, int, int] = (35, 40, 40),
    hsv_upper: tuple[int, int, int] = (85, 255, 255),
    morph_kernel_size: int = 15,
) -> np.ndarray:
    """Create a binary mask of the green playing field area.

    Args:
        frame: BGR image.
        hsv_lower: Lower HSV bound for green.
        hsv_upper: Upper HSV bound for green.
        morph_kernel_size: Kernel size for morphological closing.

    Returns:
        Binary mask (H, W) where 255 = playing field.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def filter_by_field_overlap(
    detections: list[list[float]],
    field_mask: np.ndarray,
    min_overlap: float = 0.3,
) -> list[list[float]]:
    """Keep detections whose bottom half overlaps with the playing field mask.

    Uses the bottom half of each bbox because players' feet touch the field
    while their upper body may extend above the green region.

    Args:
        detections: list of [x1, y1, x2, y2, conf].
        field_mask: Binary mask (H, W) where 255 = playing field.
        min_overlap: Minimum fraction of bottom-half pixels on field.

    Returns:
        Filtered detections.
    """
    h_mask, w_mask = field_mask.shape[:2]
    kept = []
    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        x1 = max(0, min(x1, w_mask - 1))
        x2 = max(0, min(x2, w_mask))
        y_mid = max(0, min((y1 + y2) // 2, h_mask - 1))
        y2 = max(0, min(y2, h_mask))

        roi = field_mask[y_mid:y2, x1:x2]
        if roi.size == 0:
            continue
        if np.count_nonzero(roi) / roi.size >= min_overlap:
            kept.append(det)
    return kept


def build_white_line_mask(
    frame: np.ndarray,
    white_lower: tuple[int, int, int] = (0, 0, 180),
    white_upper: tuple[int, int, int] = (180, 40, 255),
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Create a binary mask of white field line pixels.

    White lines have low saturation and high brightness in HSV.

    Args:
        frame: BGR image.
        white_lower: Lower HSV bound for white (hue ignored, low sat, high val).
        white_upper: Upper HSV bound for white.
        morph_kernel_size: Kernel size for morphological closing to join broken pixels.

    Returns:
        Binary mask (H, W) where 255 = white line pixel.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(white_lower), np.array(white_upper))
    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_field_lines(
    frame: np.ndarray,
    white_lower: tuple[int, int, int] = (0, 0, 180),
    white_upper: tuple[int, int, int] = (180, 40, 255),
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 40,
    max_line_gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Detect white field line segments using probabilistic Hough transform.

    Pipeline: white HSV mask → Canny edges → HoughLinesP.

    Args:
        frame: BGR image.
        white_lower: Lower HSV bound for white pixels.
        white_upper: Upper HSV bound for white pixels.
        canny_low: Lower threshold for Canny edge detector.
        canny_high: Upper threshold for Canny edge detector.
        hough_threshold: Minimum vote count to accept a line.
        min_line_length: Minimum segment length in pixels.
        max_line_gap: Maximum gap in pixels to join collinear segments.

    Returns:
        List of (x1, y1, x2, y2) line segments.
    """
    white_mask = build_white_line_mask(frame, white_lower, white_upper)
    edges = cv2.Canny(white_mask, canny_low, canny_high)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return []
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in raw[:, 0]]


def filter_detections(
    detections: list[list[float]],
    frame: np.ndarray | None = None,
    shape_filter_config: dict[str, Any] | None = None,
    field_mask_config: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Apply all configured filters to a list of detections.

    Args:
        detections: list of [x1, y1, x2, y2, conf].
        frame: BGR image (needed only when the playing field mask is enabled).
        shape_filter_config: ``detection_shape_filter`` section from config.yaml.
        field_mask_config: ``playing_field_mask`` section from config.yaml.

    Returns:
        Filtered detections.
    """
    if not detections:
        return []

    filtered = list(detections)

    shape_cfg = shape_filter_config or {}
    if shape_cfg.get("enabled", True):
        filtered = filter_by_size(
            filtered,
            min_w=shape_cfg.get("min_bbox_width", 20),
            min_h=shape_cfg.get("min_bbox_height", 40),
            max_w=shape_cfg.get("max_bbox_width", 300),
            max_h=shape_cfg.get("max_bbox_height", 500),
        )
        filtered = filter_by_aspect_ratio(
            filtered,
            min_ratio=shape_cfg.get("min_aspect_ratio", 0.3),
            max_ratio=shape_cfg.get("max_aspect_ratio", 4.0),
        )

    mask_cfg = field_mask_config or {}
    if mask_cfg.get("enabled", False) and frame is not None:
        field_mask = build_playing_field_mask(
            frame,
            hsv_lower=tuple(mask_cfg.get("hsv_lower", [35, 40, 40])),
            hsv_upper=tuple(mask_cfg.get("hsv_upper", [85, 255, 255])),
            morph_kernel_size=mask_cfg.get("morph_kernel_size", 15),
        )
        filtered = filter_by_field_overlap(
            filtered, field_mask, min_overlap=mask_cfg.get("min_overlap", 0.3)
        )

    return filtered
