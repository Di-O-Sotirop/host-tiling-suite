#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// === CONFIGURATION ===
// These values can be replaced automatically by the benchmark runner.
#define MIN_SIZE_KB 1
#define MAX_SIZE_MB 512
#define TOLERANCE 0.05
#define REPEATS 100

// Derived constants (do not modify directly)
constexpr size_t MIN_SIZE = MIN_SIZE_KB << 10;   // in bytes
constexpr size_t MAX_SIZE = MAX_SIZE_MB << 20;   // in bytes
constexpr double EPSILON = TOLERANCE;
constexpr size_t RESOLUTION = 1 << 10;           // Step size = 1KB

// std::cout << "[CONFIG] MIN_SIZE = " << MIN_SIZE << " bytes\n";
// std::cout << "[CONFIG] MAX_SIZE = " << MAX_SIZE << " bytes\n";
// std::cout << "[CONFIG] EPSILON  = " << EPSILON << "\n";


// Measure H2D bandwidth for a given size (bytes)
double measure_bandwidth(size_t bytes) {
    if (bytes % sizeof(float) != 0) {
        std::cerr << "[WARNING] Skipping size " << bytes
                << " (not divisible by sizeof(float) = "
                << sizeof(float) << ")\n";
        return 0.0;  // Return 0 bandwidth as a signal of invalid measurement
    }

    std::cout << "[DEBUG] Allocating " << bytes << " bytes for host and device buffers\n";

    float *h_buf = nullptr, *d_buf = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_buf, bytes));
    CHECK_CUDA(cudaMalloc(&d_buf, bytes));

    std::cout << "[DEBUG] Initializing host buffer\n";
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        h_buf[i] = static_cast<float>(i % 256);
    }

    std::cout << "[DEBUG] Performing warm-up transfer\n";
    CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "[DEBUG] Starting timed transfers\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPEATS; ++i) {
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));

    std::chrono::duration<double> elapsed = end - start;
    double total_MB = (bytes * REPEATS) / (1024.0 * 1024.0);
    double bandwidth = total_MB / elapsed.count();
    
    std::cout << "[DEBUG] Completed transfer. Bandwidth = " << bandwidth << " MB/s\n";
    return bandwidth;
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Starting bandwidth saturation detection...\n";

    // Step 1: get peak bandwidth at max size
    std::cout << "[DEBUG] Measuring peak bandwidth at MAX_SIZE = " << MAX_SIZE << " bytes\n";
    double peak_bw = measure_bandwidth(MAX_SIZE);
    std::cout << "Peak bandwidth at max size: " << peak_bw << " MB/s\n";

    // Step 2: binary search
    size_t low = MIN_SIZE, high = MAX_SIZE;
    size_t best = MAX_SIZE;

    while (high - low > RESOLUTION) {
        size_t mid = (low + high) / 2;
        std::cout << "[DEBUG] Testing transfer size: " << mid << " bytes (" << mid / 1024 << " KB)\n";
        double bw = measure_bandwidth(mid);
        std::cout << "Size: " << mid / 1024 << " KB, BW: " << bw << " MB/s\n";

        if (bw >= (1.0 - EPSILON) * peak_bw) {
            best = mid;
            high = mid;
        } else {
            low = mid + RESOLUTION;
        }
    }

    // for (int i=low; i<high; i*=2)
    // {
    //     std::cout << "[DEBUG] Testing transfer size: " << i << " bytes (" << i / 1024 << " KB)\n";
    //     double bw = measure_bandwidth(i);
    //     std::cout << "Size: " << i / 1024 << " KB, BW: " << bw << " MB/s\n"; 
    //     best = bw;
    // }    
    
    std::cout << "\nSaturation point found: " << best / 1024 << " KB (" << best << " bytes)\n";
    std::cout << "Measured BW: " << measure_bandwidth(best) << " MB/s\n";
    return 0;
}
