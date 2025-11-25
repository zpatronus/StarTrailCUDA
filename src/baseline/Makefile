# Makefile for baseline and tests

CXX := g++
CXXFLAGS := -std=c++17 -O2

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# Build directory for temporary files and binaries
BUILD_DIR := build

# Sources in current directory and subdirectories
SRCS := $(wildcard *.cpp) $(wildcard frame_reader/*.cpp) $(wildcard renderer/*.cpp)
TARGET := $(BUILD_DIR)/startrail_baseline
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Test sources in `test/` (place test objects directly under $(BUILD_DIR) to avoid build/test/)
TEST_SRCS := $(wildcard test/*.cpp)
TEST_TARGET := $(BUILD_DIR)/test_opencv
TEST_OBJS := $(patsubst test/%.cpp,$(BUILD_DIR)/%.o,$(TEST_SRCS))

# Sampler sources in `sampler/`
SAMPLER_SRCS := $(wildcard sampler/*.cpp)
SAMPLER_TARGET := $(BUILD_DIR)/video_sampler
SAMPLER_OBJS := $(patsubst sampler/%.cpp,$(BUILD_DIR)/sampler_%.o,$(SAMPLER_SRCS))

FORMAT_FILES := $(shell find . -maxdepth 2 -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' \) -print)

.PHONY: all test clean format check-format sampler

# Run formatting check before building; if check-format fails, make will stop.
all: check-format $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $^ $(OPENCV_LIBS) -lsqlite3

test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $^ $(OPENCV_LIBS) -lsqlite3

sampler: $(SAMPLER_TARGET)

$(SAMPLER_TARGET): $(SAMPLER_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $^ $(OPENCV_LIBS) -lsqlite3


$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Compile frame_reader sources
$(BUILD_DIR)/frame_reader/%.o: frame_reader/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Compile renderer sources
$(BUILD_DIR)/renderer/%.o: renderer/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Compile test sources located under test/ into build/<name>.o (avoid build/test/)
$(BUILD_DIR)/%.o: test/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Compile sampler sources located under sampler/ into build/sampler_<name>.o
$(BUILD_DIR)/sampler_%.o: sampler/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

# Apply clang-format in-place to project C/C++ sources under this directory
format:
	@command -v clang-format >/dev/null 2>&1 || (echo "clang-format not found. Install it and retry."; exit 1)
	@echo "Formatting files..."
	@for f in $(FORMAT_FILES); do \
		clang-format -i "$$f"; \
	done
	@echo "Formatting complete."

# Check formatting; exit non-zero if any file differs from clang-format --style=file output
check-format:
	@command -v clang-format >/dev/null 2>&1 || (echo "clang-format not found. Install it and retry."; exit 1)
	@echo "Checking formatting..."
	@DIFF=0; for f in $(FORMAT_FILES); do \
			clang-format --style=file "$$f" | diff -u "$$f" - >/dev/null || { echo "Formatting difference: $$f"; DIFF=1; }; \
		done; \
		if [ $$DIFF -ne 0 ]; then echo "Formatting issues found. Please run make format."; exit 1; else echo "OK"; fi
