## CUDA Star Trail Renderer

CUDA-accelerated video star trail effect renderer with async producer-consumer pipeline.

### Features

- **Pure CUDA Implementation**: No OpenCV dependency, uses CUDA for all processing
- **Hardware Video Decode/Encode**: Uses FFmpeg's libavcodec for video I/O
- **Async Pipeline**: Producer-consumer architecture with separate threads for decode, render, and encode
- **Multiple Algorithms**: MAX, AVERAGE, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY
- **Same CLI as Baseline**: Drop-in replacement for the baseline implementation

### Architecture

```
┌─────────┐      ┌──────────┐      ┌─────────┐
│ Decoder │─────▶│ Renderer │─────▶│ Encoder │
│ Thread  │Queue │  Thread  │Queue │ Thread  │
└─────────┘      └──────────┘      └─────────┘
```

- **Decoder**: Reads video with FFmpeg, converts to RGB, uploads to GPU memory
- **Renderer**: Applies star trail effects using CUDA kernels on GPU
- **Encoder**: Downloads from GPU, converts to YUV420P, encodes with FFmpeg

### Prerequisites

1. **CUDA Toolkit** (11.0 or later)
```bash
# Check CUDA installation
nvcc --version
```

2. **FFmpeg development libraries**
```bash
sudo apt update
sudo apt install -y libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
```

3. **NVIDIA GPU** with compute capability 7.5 or higher

### Build

```bash
cd src/cuda
make
```

The executable will be created at `build/startrail_cuda`.

### Usage

```bash
./build/startrail_cuda -i INPUT -o OUTPUT -a ALGORITHM [OPTIONS]

Required:
  -i, --input VIDEO_PATH      Input video file path
  -o, --output VIDEO_PATH     Output video file path
  -a, --algorithm ALGORITHM   Render algorithm: MAX, AVERAGE, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY

Optional:
  -f, --fps FPS               Output video fps (default: input video fps)
  -s, --step STEP             Input video sampling step (default: 1)
  -w, --window-size SIZE      Window size for AVERAGE, LINEAR, LINEARAPPROX (default: fps)
  -h, --help                  Show help message
```

### Examples

```bash
# MAX algorithm (star trails with decay)
./build/startrail_cuda -i input.mp4 -o output.mp4 -a MAX

# LINEAR APPROX with custom window size
./build/startrail_cuda -i input.mp4 -o output.mp4 -a LINEARAPPROX -w 60

# AVERAGE with custom fps and step
./build/startrail_cuda -i input.mp4 -o output.mp4 -a AVERAGE -w 12 -f 30 -s 2
```

### Algorithm Details

- **MAX**: Takes maximum pixel value over time with exponential decay (0.95)
- **AVERAGE**: Sliding window average of recent frames
- **EXPONENTIAL**: Exponential moving average with factor 0.05
- **LINEAR**: Weighted max with linearly decaying weights
- **LINEARAPPROX**: Approximates linear decay using exponential function
- **DUMMY**: Pass-through (no effect)

### Performance Tips

- Use `-s 2` or higher to process every Nth frame for faster processing
- Larger `-w` values create longer star trails but use more GPU memory
- The async pipeline overlaps decode, render, and encode for maximum throughput

### Clean

```bash
make clean
```

### Troubleshooting

**CUDA errors**: Ensure you have a compatible NVIDIA GPU and drivers installed
```bash
nvidia-smi
```

**FFmpeg errors**: Check that all FFmpeg libraries are installed
```bash
pkg-config --modversion libavformat libavcodec libavutil
```

**Compute capability**: Update `-arch=sm_75` in Makefile to match your GPU
- RTX 20xx/30xx/40xx: `sm_75` or `sm_86` or `sm_89`
- GTX 10xx: `sm_61`
- Check your GPU: https://developer.nvidia.com/cuda-gpus
