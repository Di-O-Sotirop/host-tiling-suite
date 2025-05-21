# Compiler and flags
NVCC := nvcc
CFLAGS := -O3 -Xcompiler -fopenmp

# Sources and executables
# BASELINE_SRC := src/gemver_AK1_tiled_baseline.cu
# TILED_SRC    := src/gemver_AK1_tiled.cu

# BASELINE_EXE := gemver_AK1_tiled_baseline.exe
# TILED_EXE    := gemver_AK1_tiled.exe
BASELINE_SRC := src/gemver_AK1_tiled_baseline_fine.cu
TILED_SRC    := src/gemver_AK1_tiled_fine.cu

BASELINE_EXE := gemver_AK1_tiled_baseline.exe
TILED_EXE    := gemver_AK1_tiled.exe

# Optional define injection
ifdef VERIFY
  VERIFY_FLAG := -DVERIFY
endif

# Build rules
all: $(BASELINE_EXE) $(TILED_EXE)

$(BASELINE_EXE): $(BASELINE_SRC)
	$(NVCC) $(CFLAGS) $(VERIFY_FLAG) -o $@ $<

$(TILED_EXE): $(TILED_SRC)
	$(NVCC) $(CFLAGS) $(VERIFY_FLAG) -o $@ $<

clean:
	rm -f $(BASELINE_EXE) $(TILED_EXE)
