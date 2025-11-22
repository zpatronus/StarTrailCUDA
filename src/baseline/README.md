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
./test_opencv
```
