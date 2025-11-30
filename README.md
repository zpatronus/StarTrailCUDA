## Install opencv

For Ubuntu:

1. Download Opencv

```bash
sudo apt update
sudo apt install -y libopencv-dev
```

2. Verify installation

```bash
pkg-config --modversion opencv4 #Should print 4.2.0 / 4.5.x / 4.6.x
```

3. Download sqlite3 and gdal

```bash
sudo apt install -y libsqlite3-dev
sudo apt install -y libgdal-dev
```

## Compile and run

```bash
make
./build/startrail_baseline -i input.mp4 -o output.mp4 -a ALGORITHM
# For video sampler
make sampler
./build/video_sampler
```

### Usage

```bash
./build/startrail_baseline -i INPUT -o OUTPUT -a ALGORITHM [OPTIONS]

Required:
  -i, --input VIDEO_PATH      Input video file path
  -o, --output VIDEO_PATH     Output video file path
  -a, --algorithm ALGORITHM   Render algorithm: MAX, AVERAGE, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY

Optional:
  -f, --fps FPS               Output video fps (default: input video fps)
  -s, --step STEP             Input video sampling step (default: 1)
  -w, --window-size SIZE      Window size for AVERAGE, LINEAR, LINEARAPPROX (default: fps)
  -h, --help                  Show help message

Examples:
  ./build/startrail_baseline -i input.mp4 -o output.mp4 -a MAX
  ./build/startrail_baseline -i input.mp4 -o output.mp4 -a LINEARAPPROX -w 60
  ./build/startrail_baseline -i input.mp4 -o output.mp4 -a AVERAGE -w 12 -f 30 -s 2
```

## For Developers

```bash
sudo apt install -y clang-format # Install clang-format
pip install --user pre-commit # Install pre-commit
cd StarTrailCUDA # Switch to project directory
pre-commit install # Enable pre-commit
```
