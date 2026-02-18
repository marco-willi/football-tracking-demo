#!/usr/bin/env bash
# Download football match clip from YouTube (CC-BY licensed)
#
# Video: Amateur Football Match
# Source: https://www.youtube.com/watch?v=Rq4QD4vKjz8
# License: Creative Commons Attribution (CC-BY)
# Clip: 35:18 - 36:20 (~62 seconds)
#
# Requires: yt-dlp (pip install yt-dlp)

set -euo pipefail

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_URL="https://www.youtube.com/watch?v=Rq4QD4vKjz8"
START_TIME="00:35:18"
END_TIME="00:36:20"
OUTPUT_DIR="${PROJECT_ROOT}/data"
OUTPUT_FILE="${OUTPUT_DIR}/match.mp4"

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "Error: yt-dlp is not installed. Install with: pip install yt-dlp"
    exit 1
fi

# Check if ffmpeg is installed (needed for clipping)
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Install with: apt-get install ffmpeg"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Skip if already downloaded
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Video already exists at ${OUTPUT_FILE}, skipping download."
    echo "Delete the file and re-run to download again."
    exit 0
fi

echo "Downloading clip from ${VIDEO_URL}"
echo "Time range: ${START_TIME} - ${END_TIME}"
echo ""

# Download and clip in one step using yt-dlp's built-in support
# --download-sections: extract time range
# -f: select best mp4 format up to 720p
yt-dlp \
    --download-sections "*${START_TIME}-${END_TIME}" \
    -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]" \
    --merge-output-format mp4 \
    --force-keyframes-at-cuts \
    -o "${OUTPUT_FILE}" \
    "${VIDEO_URL}"

echo ""
echo "Download complete: ${OUTPUT_FILE}"
echo ""

# Print video info
if command -v ffprobe &> /dev/null; then
    echo "Video info:"
    ffprobe -v quiet -show_entries format=duration -show_entries stream=width,height,r_frame_rate \
        -of default=noprint_wrappers=1 "${OUTPUT_FILE}" 2>/dev/null | head -5
fi
