#!/usr/bin/env python3
"""Main entry point for the football player tracking demo pipeline.

Usage:
    python -m football_tracking_demo.run_demo --video data/match.mp4
    python -m football_tracking_demo.run_demo --config config/config.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import imageio
from tqdm import tqdm

from football_tracking_demo.config import (
    get_detection_config,
    get_detection_shape_filter_config,
    get_hud_mask_config,
    get_output_config,
    get_playing_field_mask_config,
    get_tracking_config,
    get_visualization_config,
    load_config,
)
from football_tracking_demo.detector import PlayerDetector
from football_tracking_demo.tracker import build_tracker
from football_tracking_demo.video_io import VideoWriter, get_video_metadata, load_video
from football_tracking_demo.viz import TrackVisualizer


def save_tracks_jsonl(tracks_data: list[dict[str, Any]], output_path: str) -> None:
    """Save track data to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in tracks_data:
            f.write(json.dumps(record) + "\n")


def create_demo_gif(
    video_path: str,
    output_path: str,
    start_frame: int = 0,
    duration_seconds: float = 5.0,
    target_width: int = 720,
) -> None:
    """Create a demo GIF from the annotated video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(duration_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = target_width / w
        frames.append(cv2.resize(frame_rgb, (target_width, int(h * scale))))

    cap.release()

    if frames:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, frames, fps=min(fps, 15), loop=0)
        print(f"GIF saved to {output_path} ({len(frames)} frames)")


def compute_track_metrics(all_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute basic tracking metrics."""
    if not all_tracks:
        return {"total_detections": 0, "unique_tracks": 0}

    frames_with_tracks: dict[int, int] = {}
    track_lengths: dict[int, int] = {}

    for record in all_tracks:
        frame = record["frame"]
        tid = record["track_id"]
        frames_with_tracks[frame] = frames_with_tracks.get(frame, 0) + 1
        track_lengths[tid] = track_lengths.get(tid, 0) + 1

    n_frames = len(frames_with_tracks)
    n_tracks = len(track_lengths)

    return {
        "total_detections": len(all_tracks),
        "unique_tracks": n_tracks,
        "total_frames_with_tracks": n_frames,
        "avg_tracks_per_frame": round(sum(frames_with_tracks.values()) / n_frames, 2)
        if n_frames
        else 0,
        "avg_track_length_frames": round(sum(track_lengths.values()) / n_tracks, 2)
        if n_tracks
        else 0,
        "track_lengths": track_lengths,
    }


def plot_track_metrics(metrics: dict[str, Any], output_path: str) -> None:
    """Generate and save a simple metrics plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    track_lengths = list(metrics.get("track_lengths", {}).values())
    if track_lengths:
        axes[0].hist(track_lengths, bins=20, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Track Length (frames)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Track Lengths")

    summary_text = (
        f"Total Detections: {metrics['total_detections']}\n"
        f"Unique Tracks: {metrics['unique_tracks']}\n"
        f"Avg Tracks/Frame: {metrics['avg_tracks_per_frame']}\n"
        f"Avg Track Length: {metrics['avg_track_length_frames']} frames"
    )
    axes[1].text(
        0.5,
        0.5,
        summary_text,
        transform=axes[1].transAxes,
        fontsize=14,
        va="center",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )
    axes[1].axis("off")
    axes[1].set_title("Tracking Summary")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Metrics plot saved to {output_path}")


def run_pipeline(config: dict[str, Any], video_path: str | None = None) -> None:
    """Run the full detection and tracking pipeline."""
    det_cfg = get_detection_config(config)
    hud_cfg = get_hud_mask_config(config)
    shape_cfg = get_detection_shape_filter_config(config)
    mask_cfg = get_playing_field_mask_config(config)
    track_cfg = get_tracking_config(config)
    viz_cfg = get_visualization_config(config)
    out_cfg = get_output_config(config)

    input_path = video_path or config.get("video", {}).get(
        "input_path", "data/match.mp4"
    )
    print(f"Input video: {input_path}")

    metadata = get_video_metadata(input_path)
    fps, width, height = metadata["fps"], metadata["width"], metadata["height"]
    frame_count = metadata["frame_count"]
    print(f"Video: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")

    detector = PlayerDetector(
        model_name=det_cfg.get("model", "yolov8n.pt"),
        model_dir=det_cfg.get("model_dir", "checkpoints"),
        conf_threshold=det_cfg.get("confidence_threshold", 0.4),
        iou_threshold=det_cfg.get("nms_iou_threshold", 0.45),
        device=det_cfg.get("device", "cpu"),
        hud_top=hud_cfg.get("top_percent", 0.10),
        hud_bottom=hud_cfg.get("bottom_percent", 0.12),
        hud_enabled=hud_cfg.get("enabled", True),
        shape_filter_config=shape_cfg,
        field_mask_config=mask_cfg,
    )

    tracker = build_tracker(track_cfg)
    print(f"Tracker: {track_cfg.get('tracker', 'bytetrack')}")
    visualizer = TrackVisualizer.from_config(viz_cfg)

    output_video_path = out_cfg.get("video_path", "outputs/demo.mp4")
    output_tracks_path = out_cfg.get("tracks_path", "outputs/tracks.jsonl")
    output_metrics_path = out_cfg.get("metrics_path", "outputs/track_stats.png")
    output_gif_path = out_cfg.get("gif_path", "outputs/demo.gif")

    all_tracks: list[dict[str, Any]] = []

    with VideoWriter(output_video_path, fps, width, height) as writer:
        pbar = tqdm(
            enumerate(load_video(input_path)), total=frame_count, desc="Processing"
        )

        for frame_idx, frame in pbar:
            timestamp = frame_idx / fps
            detections = detector.detect_and_filter(frame)
            tracks = tracker.update(detections, frame)

            for trk in tracks:
                all_tracks.append(
                    {
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "track_id": int(trk[4]),
                        "bbox": [
                            round(trk[0], 1),
                            round(trk[1], 1),
                            round(trk[2], 1),
                            round(trk[3], 1),
                        ],
                        "confidence": round(trk[5], 3),
                    }
                )

            writer.write(visualizer.draw(frame, tracks))
            pbar.set_postfix({"tracks": len(tracks)})

    print(f"\nAnnotated video saved to {output_video_path}")

    save_tracks_jsonl(all_tracks, output_tracks_path)
    print(f"Track data saved to {output_tracks_path} ({len(all_tracks)} records)")

    metrics = compute_track_metrics(all_tracks)
    print("\nTracking metrics:")
    print(f"  Unique tracks: {metrics['unique_tracks']}")
    print(f"  Avg tracks/frame: {metrics['avg_tracks_per_frame']}")
    print(f"  Avg track length: {metrics['avg_track_length_frames']} frames")

    plot_track_metrics(metrics, output_metrics_path)

    create_demo_gif(
        output_video_path,
        output_gif_path,
        out_cfg.get("gif_start_frame", 150),
        out_cfg.get("gif_duration", 5),
        out_cfg.get("gif_width", 720),
    )

    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="Football Player Tracking Demo")
    parser.add_argument(
        "--video", type=str, help="Path to input video file (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, args.video)


if __name__ == "__main__":
    main()
