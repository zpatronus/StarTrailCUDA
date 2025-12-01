# StarTrailCUDA: GPU-Accelerated Rendering of Night-Sky Star Trails Video

![StarTrail Example](./docs/project_proposal.assets/image-20251117192727072.png)

**Team Members:** Zijun Yang <zijuny@andrew.cmu.edu> and Jiache Zhang <jiachez@andrew.cmu.edu>

StarTrailCUDA is a GPU-accelerated rendering pipeline that converts a large sequence of fixed-camera night-sky frames into stunning star-trail time-lapse videos using CUDA. 

## Environment

The project has been verified to work on AWS EC2 g4dn.xlarge instance, but should work on any Ubuntu machine with NVIDIA GPU support.

## Setup

The setup process is simple:

1. Run `./setup-aws.sh`
2. Reboot your machine
3. You're done!

## Video Download

To download videos, check out `src/video_download`. You might want to change the encoding from AV1 to H264 for better compatibility.

## Build and Run

To build and run implementations, navigate to the respective folders in `src` and run `make`:

```bash
cd src/baseline
make
```

The implementations have a similar argument list. Here are examples using the CUDA implementation:

**Running MAX algorithm:**

```bash
./build/startrail_cuda -i ../video_download/test_starrail_15sec_h264.mp4 -o output.mp4 -a MAX
```

**Running LINEARAPPROX algorithm with window size specified:**

```bash
./build/startrail_cuda -i ../video_download/test_starrail_15sec_h264.mp4 -o output.mp4 -a LINEARAPPROX -w 60
```

The main arguments are:

- `-i`: input video path
- `-o`: output video path
- `-a`: algorithm (MAX, LINEARAPPROX, etc.)
- `-w`: window size (for algorithms that support it)

The test video used in the examples is `test_starrail_15sec_h264.mp4`.

## For Developers

```bash
sudo apt install -y clang-format # Install clang-format
pip install --user pre-commit # Install pre-commit
cd StarTrailCUDA # Switch to project directory
pre-commit install # Enable pre-commit
```
