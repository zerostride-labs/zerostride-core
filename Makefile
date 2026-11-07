CXX      ?= c++
AR       ?= ar
UNAME_M  := $(shell uname -m)

WARN_FLAGS := -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor
CXXFLAGS   := -std=c++20 -O3 $(WARN_FLAGS) -fno-exceptions -fno-rtti -Iinclude -Isrc

ifeq ($(UNAME_M),x86_64)
	CXXFLAGS += -mavx2 -mfma -mf16c
endif

ifeq ($(SANITIZE),1)
	CXXFLAGS += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
	LDFLAGS  += -fsanitize=address,undefined
endif

BUILD_DIR  := build
LIB_TARGET := $(BUILD_DIR)/libzerostride_core.a
BENCH_BIN  := $(BUILD_DIR)/zs-bench
TEST_BIN   := $(BUILD_DIR)/test_telemetry

LIB_SRCS := \
	src/platform/os_resources.cpp \
	src/core/telemetry.cpp

LIB_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

.PHONY: all test sanitize bench clean

all: $(LIB_TARGET) $(BENCH_BIN) $(TEST_BIN)

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(LIB_OBJS)
	$(AR) rcs $@ $^

$(BENCH_BIN): tools/zs-bench/main.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TEST_BIN): tests/test_telemetry.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

sanitize:
	$(MAKE) clean
	$(MAKE) test SANITIZE=1
	$(MAKE) clean

bench: $(BENCH_BIN)
	./$(BENCH_BIN) --steps 128 --buffer-mb 16 --mode arena --label "arena-pinned"
	./$(BENCH_BIN) --steps 128 --buffer-mb 16 --mode fault --label "demand-paged-fault"

clean:
	rm -rf $(BUILD_DIR)
