#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -o OUTPUT_DIR -n FILENAME -i VIDEO_ID_OR_URL

Options:
  -o OUTPUT_DIR        Directory to save the downloaded file (will be created)
  -n FILENAME          Output filename without extension (e.g. "input")
  -i VIDEO_ID_OR_URL   YouTube video id (the part after v=) or a full URL
  -h                   Show this help message

Example:
  $(basename "$0") -o /tmp -n input -i Bbp1-p2FoXU
  # saves as /tmp/input.mp4 (or input.m4a depending on stream)

EOF
}

OUTPUT_DIR=""
FILENAME=""
VIDEO=""

while getopts "o:n:i:h" opt; do
  case "$opt" in
    o) OUTPUT_DIR="$OPTARG" ;;
    n) FILENAME="$OPTARG" ;;
    i) VIDEO="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ -z "$OUTPUT_DIR" ] || [ -z "$FILENAME" ] || [ -z "$VIDEO" ]; then
  echo "Error: missing required argument."
  usage
  exit 1
fi

# Ensure yt-dlp is available
if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "yt-dlp not found in PATH. Install it (recommended: pipx install yt-dlp) or ensure it's available." >&2
  exit 2
fi

# Create output dir if necessary
mkdir -p "$OUTPUT_DIR"

# Build URL if user provided only an id
if [[ "$VIDEO" =~ ^https?:// ]]; then
  URL="$VIDEO"
else
  URL="https://www.youtube.com/watch?v=$VIDEO"
fi

OUT_TEMPLATE="$OUTPUT_DIR/$FILENAME.%(ext)s"

echo "Downloading $URL -> $OUT_TEMPLATE"

# Run yt-dlp with chosen format
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" \
  -o "$OUT_TEMPLATE" \
  "$URL"

ret=$?
if [ $ret -eq 0 ]; then
  echo "Download finished."
else
  echo "yt-dlp exited with code $ret" >&2
fi

exit $ret
