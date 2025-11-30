CXX := g++
CXXFLAGS := -std=c++17 -O2

# OpenCV flags
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# FFmpeg flags
FFMPEG_CFLAGS := $(shell pkg-config --cflags libavformat libavcodec libavutil libswscale)
FFMPEG_LIBS   := $(shell pkg-config --libs libavformat libavcodec libavutil libswscale)

# Build directory
BUILD_DIR := build

# Source files
SRCS := $(wildcard *.cpp) $(wildcard frame_reader/*.cpp) $(wildcard renderer/*.cpp)
TARGET := $(BUILD_DIR)/startrail_baseline_nvenc
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

FORMAT_FILES := $(shell find . -maxdepth 2 -type f \( -name '*.cpp' -o -name '*.h' \) -print)

.PHONY: all clean format check-format

all: check-format $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS) $(FFMPEG_LIBS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

# Compile frame_reader sources
$(BUILD_DIR)/frame_reader/%.o: frame_reader/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

# Compile renderer sources
$(BUILD_DIR)/renderer/%.o: renderer/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS) -c $< -o $@

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
