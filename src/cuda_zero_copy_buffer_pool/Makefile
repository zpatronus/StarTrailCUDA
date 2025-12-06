CXX = g++
NVCC = nvcc

CUDA_PATH ?= /usr/local/cuda
NVIDIA_VIDEO_CODEC_PATH ?= /usr/local/nvidia-video-codec

CXXFLAGS = -std=c++17 -O3 -Wall -I$(CUDA_PATH)/include
NVCCFLAGS = -std=c++17 -O3 -arch=sm_75

LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lnvcuvid -lnvidia-encode -lavformat -lavcodec -lavutil -lpthread

TARGET = build/startrail_zero_copy
SOURCES = main.cpp \
          buffer_pool.cpp \
          frame_ref_queue.cpp \
          decoder.cpp \
          encoder.cpp \
          renderer.cu \
          utils/cuda_utils.cu

OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -rf build

format:
	clang-format -i *.cpp *.h *.cu utils/*.h utils/*.cu

.PHONY: all clean format

