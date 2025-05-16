NVCC := nvcc
CFLAGS := -O3

SRC_DIR := src
BIN := .

BASELINE_SRC := $(SRC_DIR)/vec_add3_async.cu
TILED_SRC := $(SRC_DIR)/vec_add3_tiled.cu

BASELINE_EXE := $(BIN)/vec_add3_async.exe
TILED_EXE := $(BIN)/vec_add3_tiled.exe

all: $(BASELINE_EXE) $(TILED_EXE)

$(BASELINE_EXE): $(BASELINE_SRC)
	$(NVCC) $(CFLAGS) $< -o $@

$(TILED_EXE): $(TILED_SRC)
	$(NVCC) $(CFLAGS) $< -o $@

baseline: $(BASELINE_EXE)
tiled: $(TILED_EXE)

clean:
	rm -f $(BASELINE_EXE) $(TILED_EXE)
