# Football Player Tracking Demo

Detect and track football (soccer) players in broadcast video using YOLO26 (or YOLOv8) and ByteTrack. The pipeline processes a short match clip frame-by-frame, assigns stable IDs across frames, and exports an annotated video, structured track data, and a demo GIF.

![Demo GIF](outputs/demo.gif)

## Pipeline

```
Input Video
    |
    v
HUD Masking          -- black out top / bottom horizontals -- DISABLED (too rigid)
    |
    v
YOLO Detection       -- COCO-pretrained person detector (YOLO26 / YOLOv8)
    |
    v
Playing Field Filter -- reject detections by size, aspect ratio, green field mask
    |
    v
ByteTrack            -- multi-object tracking with stable IDs
    |
    v
Visualization        -- bounding boxes, track IDs, motion trails
    |
    v
Export               -- annotated MP4, JSONL tracks, metrics plot, demo GIF
```

### HUD masking

HUD masking removes the top X% and bottom Y% of each frame before detection, eliminating false positives of non-players. I ultimately disabled it due to it's rigidity, particularly, when camera moves it is possible no edge remains and the filter removes parts of the playing field.

<img src="outputs/hud_mask_example.png" width="700" alt="HUD Masking" />

### Playing Field Filter

The playing field (pitch) filter is used to filter detections from non-players. It uses HSV color-space to filter out regions that do not belong to the field ([HSV for green](https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv)). Note that the score display is filtered quite well, as is the right side of the field. Problematic is generally the area between the actual playing field and the end of the green.

<img src="outputs/field_mask_samples.png" width="700" alt="Hand-Tuned Selection for HSV" />

Also used is a morphological operation to 'fill' gaps in the green using differnt kernel sizes. See cClosing gaps in masks: [Link](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#closing).


<img src="outputs/field_mask_kernel_comparison.png" width="700" alt="Kernel Comparison" />

Other filters operate directly on the bbx / detection size.


### Model Variants

Model sizes range from very small (2.4M parameters) to rather arge (65 M parameters) and it has important practical implications on inference speed:

<img src="outputs/model_speed_comparison.png" width="700" alt="Inference Speed" />


Assessing the quality, without labelled data, is difficult. It seems larger models are generally better. Important is that model-size influences detection confidence, which needs to be considered when tuning the tracker.

### Byte Track

**ByteTrack** (via the [supervision](https://github.com/roboflow/supervision) library) handles online multi-object association with a 120-frame track buffer for handling brief occlusions. This could be tuned further or improved with methods based on appearaace embeddings for example.

Another issue is the kalman filter in the tracker, wich does not account for camera movements. Global movement should be corrected for.

## Project Structure

```
football-tracking-demo/
├── config/
│   └── config.yaml          # All pipeline parameters
├── src/football_tracking_demo/
│   ├── run_demo.py           # Main pipeline entry point
│   ├── config.py             # YAML config loader
│   ├── video_io.py           # Video read/write (OpenCV)
│   ├── detector.py           # YOLOv8 detection + HUD masking
│   ├── filtering.py          # Size, aspect ratio, playing field filters
│   ├── tracker.py            # ByteTrack wrapper
│   └── viz.py                # Drawing boxes, IDs, motion trails
├── notebooks/
│   ├── 01-mw-detection-hud-mask-inspection.ipynb
│   ├── 02-mw-detection-threshold.ipynb
│   ├── 03-mw-compare-models.ipynb
│   ├── 04-mw-playing-field-masking.ipynb
│   └── 05-mw-tracker-tuning.ipynb
├── scripts/
│   └── download_video.sh     # Download CC-BY source clip
├── checkpoints/              # Cached model weights (gitignored)
├── data/                     # Input video (gitignored)
├── outputs/                  # Generated artifacts
└── Makefile                  # Build automation
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Download the source video (requires yt-dlp + ffmpeg)
make data

# Run the full pipeline
make demo
```

Or run directly:

```bash
python -m football_tracking_demo.run_demo --video data/match.mp4
```

### Outputs

| File | Description |
|------|-------------|
| `outputs/demo.mp4` | Annotated video with bounding boxes and track IDs |
| `outputs/demo.gif` | 5-second GIF clip for quick preview |
| `outputs/tracks.jsonl` | One JSON record per detection per frame |
| `outputs/track_stats.png` | Track length histogram and summary statistics |

### Track data format

Each line in `tracks.jsonl`:

```json
{"frame": 125, "timestamp": 4.16, "track_id": 7, "bbox": [x1, y1, x2, y2], "confidence": 0.91}
```

## Configuration

All parameters live in [`config/config.yaml`](config/config.yaml).

### `detection`

Controls which YOLO model is used and how it runs inference.

```yaml
detection:
  model: "yolo26x.pt"       # Model weights file (see Model Variants table below)
  model_dir: "checkpoints"  # Directory where weights are cached after first download
  confidence_threshold: 0.10 # Minimum score to keep a detection (lower = more recalls)
  nms_iou_threshold: 0.45   # IoU threshold for non-max suppression (YOLOv8 only)
  class_filter: [0]         # COCO class IDs to keep; 0 = person
  device: "cuda"            # Inference device: "cpu", "cuda", or "mps"
```

### `hud_mask`

Blacks out broadcast UI regions (scoreboard, minimap) before detection to eliminate false positives that would otherwise be picked up in those areas.

```yaml
hud_mask:
  enabled: true
  top_percent: 0.05     # Fraction of frame height to black out at the top
  bottom_percent: 0.10  # Fraction of frame height to black out at the bottom
```

### `detection_shape_filter`

Rejects detections based on bounding box geometry. This is a fast, parameter-free filter applied before the more expensive colour-mask step.

```yaml
detection_shape_filter:
  enabled: true
  min_bbox_width: 10    # Pixels — drops tiny noise blobs
  min_bbox_height: 15   # Pixels — drops tiny noise blobs
  max_bbox_width: 400   # Pixels — drops very wide detections (crowd rows, banners)
  max_bbox_height: 600  # Pixels — drops very tall detections
  min_aspect_ratio: 0.2 # height/width — drops extremely wide boxes
  max_aspect_ratio: 5.0 # height/width — drops extremely tall/thin boxes
```

### `playing_field_mask`

Builds an HSV-based binary mask of the green pitch and rejects any detection whose bottom half does not overlap it sufficiently. This is the strongest filter for removing crowd and off-field detections.

```yaml
playing_field_mask:
  enabled: true
  hsv_lower: [35, 40, 40]   # Lower HSV bound for green (hue 35 ≈ yellow-green)
  hsv_upper: [85, 255, 255] # Upper HSV bound for green (hue 85 ≈ cyan-green)
  morph_kernel_size: 15     # Morphological closing kernel — larger fills bigger holes
                            # in the mask but also expands edges onto non-field areas
  min_overlap: 0.3          # Minimum fraction of the bottom-half bbox pixels that must
                            # fall on the field mask to keep the detection
```

> Tune these parameters interactively in [`04-mw-playing-field-masking.ipynb`](notebooks/04-mw-playing-field-masking.ipynb).

> To understand how `track_buffer`, `match_threshold`, and `track_activation_threshold` interact with track fragmentation, see [`05-mw-tracker-tuning.ipynb`](notebooks/05-mw-tracker-tuning.ipynb).

### `tracking`

Parameters passed to ByteTrack for multi-object association.

```yaml
tracking:
  track_buffer: 60             # Frames a lost track is kept alive before deletion;
                               # increase for longer occlusions
  match_threshold: 0.8         # Minimum IoU to match a detection to an existing track
  frame_rate: 60               # Source video FPS (used internally for velocity priors)
  track_activation_threshold: 0.25  # Minimum confidence for a detection to start a
                                     # new track (higher = fewer spurious tracks)
```

### `visualization`

Controls what is drawn on the annotated output video.

```yaml
visualization:
  show_boxes: true    # Draw bounding rectangles
  show_ids: true      # Overlay track ID numbers
  show_trails: true   # Draw motion trail from past N frames
  trail_length: 10    # Number of past frames to include in the trail
  box_thickness: 2    # Line width in pixels
  font_scale: 0.6     # OpenCV font scale for ID labels
  font_thickness: 2   # Stroke width for ID labels
```

### Model Variants

The pipeline supports YOLO26 and legacy YOLOv8 variants. Weights are downloaded on first run and cached in `checkpoints/`. At first glance, the performance is not that differente between YOLO26 and YOLOv8 but rather within the variants (from small to large).

#### YOLO26 — newest [source](https://docs.ultralytics.com/models/yolo26/))

COCO val, 640px input. CPU: ONNX, GPU: T4 TensorRT.

| Model | Params | FLOPs (B) | mAP50-95 | CPU (ms) | GPU (ms) | Notes |
|-------|--------|-----------|----------|----------|----------|-------|
| yolo26n | 2.4M | 5.4 | 40.9 | 38.9 | 1.7 | Nano -- fastest, least accurate |
| yolo26s | 9.5M | 20.7 | 48.6 | 87.2 | 2.5 | Small -- good speed/accuracy trade-off |
| yolo26m | 20.4M | 68.2 | 53.1 | 220.0 | 4.7 | Medium -- balanced |
| yolo26l | 24.8M | 86.4 | 55.0 | 286.2 | 6.2 | Large -- high accuracy |
| yolo26x | 55.7M | 193.9 | 57.5 | 525.8 | 11.8 | XLarge -- best accuracy |

#### YOLOv8

| Model | Params | FLOPs (B) | mAP50-95 | CPU (ms) | GPU (ms) | Notes |
|-------|--------|-----------|----------|----------|----------|-------|
| yolov8s | 11.2M | 28.6 | 44.9 | 128.4 | 1.2 | Small -- faster legacy baseline |
| yolov8x | 68.2M | 257.8 | 53.9 | 479.1 | 3.5 | XLarge -- strongest legacy model |

Select a variant by setting `detection.model` in `config.yaml`:

```yaml
detection:
  model: "yolo26l.pt"
```

Notebook [`03-mw-compare-models.ipynb`](notebooks/03-mw-compare-models.ipynb) provides a visual and quantitative comparison of all variants on the same footage, including inference speed benchmarks and a speed-vs-accuracy scatter plot.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`01-mw-detection-hud-mask-inspection`](notebooks/01-mw-detection-hud-mask-inspection.ipynb) | Visualize HUD mask position and compare detections with/without masking |
| [`02-mw-detection-threshold`](notebooks/02-mw-detection-threshold.ipynb) | Sweep confidence thresholds and shape filter presets; threshold × filter heatmap |
| [`03-mw-compare-models`](notebooks/03-mw-compare-models.ipynb) | Compare YOLO26 and YOLOv8 variants: detection quality, inference speed, confidence |
| [`04-mw-playing-field-masking`](notebooks/04-mw-playing-field-masking.ipynb) | Interactive tuning of HSV bounds, kernel size, and overlap threshold for the field mask |

## Makefile Targets

```
make data             # Download source video
make demo             # Run full pipeline (video + GIF + tracks + metrics)
make video            # Generate annotated video only
make tracks           # Generate JSONL track export only
make gif              # Generate demo GIF only
make tests            # Run pytest suite
make clean-outputs    # Remove generated outputs
make all              # Download data + run full pipeline
```

## Tech Stack

- **Python 3.14+**
- **YOLO26 / YOLOv8** (Ultralytics) -- object detection
- **ByteTrack** (supervision) -- multi-object tracking
- **OpenCV** -- video I/O and image processing
- **NumPy** / **Matplotlib** -- data handling and plotting
- **imageio** -- GIF generation

## Video Source Attribution

```
Title:   Amateur Football Match
Source:  https://www.youtube.com/watch?v=Rq4QD4vKjz8
License: Creative Commons Attribution (CC-BY)
Clip:    35:18 - 36:20 (~62 seconds)
```

The raw video is **not** included in this repository. Run `make data` to download it.

## Limitations

- No re-identification across camera cuts
- No ball tracking
- No jersey / team color recognition
- No top-down field mapping (homography)
- Track IDs may switch during heavy occlusion

## Development Notes

AI-assisted development was used for faster iteration; final architecture and code were reviewed and owned by me.

## License

MIT
