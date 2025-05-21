#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    std::vector<size_t> sizes;
    for (int i = 10; i <= 28; ++i)  // 1 KB to 256 MB
        sizes.push_back(1ULL << i);

    const int repeats = 50;

    for (size_t size : sizes) {
        float *h_buf, *d_buf;
        CHECK_CUDA(cudaMallocHost(&h_buf, size));
        CHECK_CUDA(cudaMalloc(&d_buf, size));

        // Initialize host buffer
        for (size_t i = 0; i < size / sizeof(float); ++i)
            h_buf[i] = static_cast<float>(i);

        // Warm-up
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeats; ++i) {
            CHECK_CUDA(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
        double bandwidth = (size / 1e6) / ms * 1000;  // MB/s

        std::cout << "Size: " << size / 1024 << " KB, "
                  << "Time: " << ms << " ms, "
                  << "Bandwidth: " << bandwidth << " MB/s" << std::endl;

        CHECK_CUDA(cudaFree(d_buf));
        CHECK_CUDA(cudaFreeHost(h_buf));
    }

    return 0;
}
