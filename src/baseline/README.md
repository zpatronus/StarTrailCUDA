## Install opencv
For Ubuntu:

1. Download Opencv
```bash
sudo apt update
sudo apt install -y libopencv-dev
```

2. Verify installation
```bash
pkg-config --modversion opencv4 #Should print 4.2.0 / 4.5.x
```

3. Download sqlite3 and gdal
```bash
sudo apt install -y libsqlite3-dev
sudo apt install -y libgdal-dev
```

## Compile and run
```bash
make
./build/startrail_baseline
# For video sampler
make sampler
./build/video_sampler
```

## Downloading videos from youtube
```bash
pip3 install yt-dlp # Install yt-dlp
./download_youtube.sh -h # Use the script to download from youtube
# Example usage
./download_youtube.sh -o . -n test_starrail -i Bbp1-p2FoXU
```

## For Developers
```bash
sudo apt install -y clang-format # Install clang-format
pip install --user pre-commit # Install pre-commit
cd StarTrailCUDA # Switch to project directory
pre-commit install # Enable pre-commit
```