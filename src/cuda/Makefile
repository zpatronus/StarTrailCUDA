CXX := g++
NVCC := nvcc

CUDA_PATH ?= /usr/local/cuda
NVIDIA_VIDEO_CODEC_PATH ?= /usr/local/nvidia-video-codec

CUDA_CFLAGS := -I$(CUDA_PATH)/include -I$(NVIDIA_VIDEO_CODEC_PATH)/Interface
CUDA_LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lnvcuvid -lnvidia-encode

ifeq ("$(wildcard $(CUDA_PATH)/include/cuda_runtime.h)","")
CUDA_PATH := /usr
CUDA_CFLAGS := -I$(CUDA_PATH)/include
endif

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

FFMPEG_CFLAGS := $(shell pkg-config --cflags libavformat libavcodec libavutil libswscale)
FFMPEG_LIBS   := $(shell pkg-config --libs libavformat libavcodec libavutil libswscale)

CXXFLAGS := -std=c++17 -O3
NVCCFLAGS := -std=c++17 -O3 -arch=sm_75

BUILD_DIR := build

CPP_SRCS := $(wildcard *.cpp) $(wildcard frame_reader/*.cpp) $(wildcard utils/*.cpp)
CUDA_SRCS := $(wildcard renderer/*.cu) $(wildcard utils/*.cu)

ALL_SRCS := $(CPP_SRCS) $(CUDA_SRCS)

TARGET := $(BUILD_DIR)/startrail_cuda

CPP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))
OBJS := $(CPP_OBJS) $(CUDA_OBJS)

FORMAT_FILES := $(shell find . -maxdepth 2 -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' \) -print)

.PHONY: all clean format check-format

all: check-format $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS) $(OPENCV_LIBS) $(FFMPEG_LIBS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

$(BUILD_DIR)/frame_reader/%.o: frame_reader/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

$(BUILD_DIR)/utils/%.o: utils/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

$(BUILD_DIR)/renderer/%.o: renderer/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) -c $< -o $@

$(BUILD_DIR)/utils/%.o: utils/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

format:
	@command -v clang-format >/dev/null 2>&1 || (echo "clang-format not found. Install it and retry."; exit 1)
	@echo "Formatting files..."
	@for f in $(FORMAT_FILES); do \
		clang-format -i "$$f"; \
	done
	@echo "Formatting complete."

check-format:
	@command -v clang-format >/dev/null 2>&1 || (echo "clang-format not found. Install it and retry."; exit 1)
	@echo "Checking formatting..."
	@DIFF=0; for f in $(FORMAT_FILES); do \
			clang-format --style=file "$$f" | diff -u "$$f" - >/dev/null || { echo "Formatting difference: $$f"; DIFF=1; }; \
		done; \
		if [ $$DIFF -ne 0 ]; then echo "Formatting issues found. Please run make format."; exit 1; else echo "OK"; fi
