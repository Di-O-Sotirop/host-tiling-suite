# Compiler and flags
NVCC := nvcc
CFLAGS := -O3

# Paths
SRC_DIR := src
SRC := $(SRC_DIR)/bandwidth_probe.cu
TARGET := bandwidth_probe.exe

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
