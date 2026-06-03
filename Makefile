BUILD_DIR   := build
BUILD_TYPE  ?= Release
JOBS        ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: all configure build test clean rebuild run

all: build

configure:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

build: configure
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	$(BUILD_DIR)/bin/APvectorDB

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
