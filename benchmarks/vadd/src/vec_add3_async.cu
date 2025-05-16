#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define MAX_STREAMS 6
#define NUM_TILES 1
#define SHIFTS 24
#define K_LOOP 1

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)    \
                      << " at " << __FILE__ << ":" << __LINE__        \
                      << std::endl;                                   \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void vaddLX(const float* A, const float* B, const float* D, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int k = 0; k < K_LOOP; ++k) {
            C[idx] += A[idx] + B[idx] + D[idx];
            C[idx] = fmodf(C[idx], 256.0f);
        }
    }
}

void cudaWarmUp() {
    const size_t warmup_size = 1024;
    float* h_buf;
    float* d_buf;

    CHECK_CUDA(cudaMallocHost(&h_buf, warmup_size));
    CHECK_CUDA(cudaMalloc(&d_buf, warmup_size));

    for (size_t i = 0; i < warmup_size / sizeof(float); ++i) {
        h_buf[i] = static_cast<float>(i);
    }

    CHECK_CUDA(cudaMemcpy(d_buf, h_buf, warmup_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_buf, d_buf, warmup_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void compute_golden(const float* A, const float* B, const float* D, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K_LOOP; ++k) {
            C[i] += A[i] + B[i] + D[i];
            C[i] = fmodf(C[i], 256.0f);
        }
    }
}

bool verify_result(const float* C_host, const float* C_device, int N, float epsilon = 1e-5) {
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(C_host[i] - C_device[i]) > epsilon) {
            mismatches++;
        }
    }
    std::cout << "Number of mismatches: " << mismatches << std::endl;
    return mismatches == 0;
}

int main() {
    const int N = 1 << SHIFTS;
    const size_t size = N * sizeof(float);

    bool verify = false;
    if (const char* env = std::getenv("VERIFY")) {
        verify = std::atoi(env) != 0;
    }

    cudaWarmUp();

    // Allocate host memory
    float *h_A, *h_B, *h_D, *h_C_device, *h_C_golden = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_A, size));
    CHECK_CUDA(cudaMallocHost(&h_B, size));
    CHECK_CUDA(cudaMallocHost(&h_D, size));
    CHECK_CUDA(cudaMallocHost(&h_C_device, size));
    if (verify) {
        CHECK_CUDA(cudaMallocHost(&h_C_golden, size));
    }

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 256);
        h_B[i] = static_cast<float>((2 * i) % 256);
        h_D[i] = static_cast<float>((3 * i) % 256);
    }

    const int TILE_SIZE = (N + NUM_TILES - 1) / NUM_TILES;
    const size_t TILE_BYTES = TILE_SIZE * sizeof(float);

    float *d_A[NUM_TILES], *d_B[NUM_TILES], *d_D[NUM_TILES], *d_C[NUM_TILES];
    for (int i = 0; i < NUM_TILES; ++i) {
        CHECK_CUDA(cudaMalloc(&d_A[i], TILE_BYTES));
        CHECK_CUDA(cudaMalloc(&d_B[i], TILE_BYTES));
        CHECK_CUDA(cudaMalloc(&d_D[i], TILE_BYTES));
        CHECK_CUDA(cudaMalloc(&d_C[i], TILE_BYTES));
    }

    cudaStream_t streams[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    auto offload_start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_TILES; ++t) {
        int offset = t * TILE_SIZE;
        int current_tile_size = std::min(TILE_SIZE, N - offset);
        size_t current_tile_bytes = current_tile_size * sizeof(float);
        int stream_id = t % MAX_STREAMS;

        CHECK_CUDA(cudaMemcpyAsync(d_A[t], h_A + offset, current_tile_bytes, cudaMemcpyHostToDevice, streams[stream_id]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[t], h_B + offset, current_tile_bytes, cudaMemcpyHostToDevice, streams[stream_id]));
        CHECK_CUDA(cudaMemcpyAsync(d_D[t], h_D + offset, current_tile_bytes, cudaMemcpyHostToDevice, streams[stream_id]));

        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((current_tile_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
        vaddLX<<<blocksPerGrid, threadsPerBlock, 0, streams[stream_id]>>>(
            d_A[t], d_B[t], d_D[t], d_C[t], current_tile_size);

        CHECK_CUDA(cudaMemcpyAsync(h_C_device + offset, d_C[t], current_tile_bytes, cudaMemcpyDeviceToHost, streams[stream_id]));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto offload_end = std::chrono::high_resolution_clock::now();

    if (verify) {
        compute_golden(h_A, h_B, h_D, h_C_golden, N);
        verify_result(h_C_golden, h_C_device, N);
    }

    for (int i = 0; i < NUM_TILES; ++i) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_D[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }

    for (int i = 0; i < MAX_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_D));
    CHECK_CUDA(cudaFreeHost(h_C_device));
    if (verify) {
        CHECK_CUDA(cudaFreeHost(h_C_golden));
    }

    std::chrono::duration<double, std::milli> offload_elapsed = offload_end - offload_start;
    std::cout << "Execution Time: " << offload_elapsed.count() << " ms\n";

    return 0;
}
