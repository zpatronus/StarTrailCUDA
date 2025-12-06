CXX = g++
NVCC = nvcc

CUDA_PATH ?= /usr/local/cuda
NVIDIA_VIDEO_CODEC_PATH ?= /usr/local/nvidia-video-codec

CXXFLAGS = -std=c++17 -O3 -Wall -I$(CUDA_PATH)/include -I$(NVIDIA_VIDEO_CODEC_PATH)/Interface
NVCCFLAGS = -std=c++17 -O3 -arch=sm_75 -I$(NVIDIA_VIDEO_CODEC_PATH)/Interface

LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lnvcuvid -lnvidia-encode -lavformat -lavcodec -lavutil -lswscale -lpthread

TARGET = build/startrail_cuda_pipeline
SOURCES = main.cpp \
          decoder/video_decoder.cpp \
          encoder/video_encoder.cpp \
          renderer/video_renderer.cu \
          utils/cuda_utils.cu \
          utils/color_convert.cu \
          utils/frame_queue.cpp

OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -rf build

format:
	clang-format -i $(SOURCES) \
		decoder/*.h \
		encoder/*.h \
		renderer/*.h \
		utils/*.h \
		utils/*.cpp

.PHONY: all clean format
