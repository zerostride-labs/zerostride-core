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

BUILD_DIR   := build
LIB_TARGET  := $(BUILD_DIR)/libzerostride_core.a
BENCH_BIN   := $(BUILD_DIR)/zs-bench
INSPECT_BIN := $(BUILD_DIR)/zs-inspect

TEST_TELEMETRY_BIN := $(BUILD_DIR)/test_telemetry
TEST_ARENA_BIN     := $(BUILD_DIR)/test_arena
TEST_GGUF_BIN      := $(BUILD_DIR)/test_gguf_index
TEST_BINS          := $(TEST_TELEMETRY_BIN) $(TEST_ARENA_BIN) $(TEST_GGUF_BIN)

LIB_SRCS := \
	src/platform/os_resources.cpp \
	src/platform/file_io.cpp \
	src/core/telemetry.cpp \
	src/memory/static_arena.cpp \
	src/memory/alloc_guard.cpp \
	src/format/gguf_index.cpp

LIB_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

.PHONY: all test sanitize bench inspect-demo clean

all: $(LIB_TARGET) $(BENCH_BIN) $(INSPECT_BIN) $(TEST_BINS)

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(LIB_OBJS)
	$(AR) rcs $@ $^

$(BENCH_BIN): tools/zs-bench/main.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(INSPECT_BIN): tools/zs-inspect/main.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TEST_TELEMETRY_BIN): tests/test_telemetry.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TEST_ARENA_BIN): tests/test_arena.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TEST_GGUF_BIN): tests/test_gguf_index.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

test: $(TEST_BINS)
	./$(TEST_TELEMETRY_BIN)
	./$(TEST_ARENA_BIN)
	./$(TEST_GGUF_BIN)

sanitize:
	$(MAKE) clean
	$(MAKE) test SANITIZE=1
	$(MAKE) clean

bench: $(BENCH_BIN)
	./$(BENCH_BIN) --steps 128 --buffer-mb 16 --mode arena --label "arena-pinned"
	./$(BENCH_BIN) --steps 128 --buffer-mb 16 --mode fault --label "demand-paged-fault"

inspect-demo: $(INSPECT_BIN)
	./$(INSPECT_BIN) --generate-synthetic /tmp/zs_demo_qwen_moe.gguf --layers 8 --experts 32 --top-k 4 --list-tensors
	rm -f /tmp/zs_demo_qwen_moe.gguf

clean:
	rm -rf $(BUILD_DIR)
