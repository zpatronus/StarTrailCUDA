#!/bin/bash
set -e

echo "=== StarTrailCUDA Development Environment Setup ==="

# 1. Update system
echo "Step 1: Updating system packages..."
sudo apt update

# 2. Install build essentials
echo "Step 2: Installing build tools..."
sudo apt install -y build-essential git cmake pkg-config

# 3. Install NVIDIA Driver (if not installed)
echo "Step 3: Installing NVIDIA drivers..."
sudo apt install -y nvidia-utils-580-server nvidia-driver-580-server

# 4. Install CUDA Toolkit
echo "Step 4: Installing CUDA Toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# 5. Install FFmpeg with NVENC/NVDEC support
echo "Step 5: Installing FFmpeg with NVENC/NVDEC support..."
# Install FFmpeg headers (required for linking)
sudo apt install -y libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

# Install nvidia-codec-headers for CUVID support
echo "  Installing NVIDIA codec headers..."
cd /tmp
rm -rf nv-codec-headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install
cd ~

# Install ffnvcodec runtime libraries
echo "  Installing NVENC/NVDEC runtime libraries..."
sudo apt install -y libnvidia-encode-580-server libnvidia-decode-580-server

# Install NVIDIA Video Codec SDK
echo "  Installing NVIDIA Video Codec SDK..."

cd /tmp
cp ~/StarTrailCUDA/Video_Codec_Interface_13.0.19.zip .
sudo apt install -y unzip
unzip -o Video_Codec_Interface_13.0.19.zip
sudo mkdir -p /usr/local/nvidia-video-codec/Interface
sudo cp -r Interface/*.h /usr/local/nvidia-video-codec/Interface/ 2>/dev/null || true
sudo cp -r *.h /usr/local/nvidia-video-codec/Interface/ 2>/dev/null || true
cd ~
echo "  NVIDIA Video Codec SDK headers installed"


# 6. Install OpenCV (for baseline version)
echo "Step 6: Installing OpenCV..."
sudo apt install -y libopencv-dev libsqlite3-dev libgdal-dev

# 7. Install development tools
echo "Step 7: Installing development tools..."
sudo apt install -y clang-format

# 8. Setup environment variables
echo "Step 8: Setting up CUDA environment..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 9. Verify installations
echo "Step 9: Verifying installations..."
nvidia-smi
nvcc --version
pkg-config --modversion opencv4
pkg-config --modversion libavformat

echo "=== Setup Complete! ==="
echo "Please reboot the system for NVIDIA drivers to take effect:"
echo "  sudo reboot"