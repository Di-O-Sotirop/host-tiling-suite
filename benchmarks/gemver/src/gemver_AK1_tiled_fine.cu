#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define SHIFTS 56
#define NUM_TILES 1
#define TILE_BUFFERS 2
#define MAX_STREAMS 2
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define GPU_DEVICE 0
// #define VERIFY 1

#ifdef VERIFY
  #define RUN_ON_CPU
#endif

#define THREADS_PER_BLOCK 256

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void warmup_kernel(float *data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256)
        data[i] += 1.0f;
}

void warmup_cuda_runtime() {
    const int size = 256;
    float *h_buf, *d_buf;
    cudaStream_t stream;

    CHECK_CUDA(cudaMallocHost(&h_buf, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buf, size * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < size; ++i)
        h_buf[i] = static_cast<float>(i);

    CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf, size * sizeof(float), cudaMemcpyHostToDevice, stream));
    warmup_kernel<<<(size + 127) / 128, 128, 0, stream>>>(d_buf);
    CHECK_CUDA(cudaMemcpyAsync(h_buf, d_buf, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void init(int n, float *alpha, float *beta, float *A, float *u1, float *v1, float *u2, float *v2, float *w, float *x, float *y, float *z) {
    *alpha = 1.0f;
    *beta = 1.0f;
    for (int i = 0; i < n; ++i) {
        u1[i] = static_cast<float>(i % 64) / n;
        u2[i] = static_cast<float>(i % 64) / (2.0f * n);
        v1[i] = static_cast<float>((i % 64) + 1) / (4.0f * n);
        v2[i] = static_cast<float>((i % 64) + 1) / (1.5f * n);
        y[i] = static_cast<float>((i % 64) + 1) / (3.0f * n);
        z[i] = static_cast<float>((i % 64) + 1) / (5.0f * n);
        x[i] = 0.0f;
        w[i] = 0.0f;

        for (int j = 0; j < n; ++j)
            A[i * n + j] = static_cast<float>((i % 64) * (j % 64));
    }
}

void compareResults(int n, const float *w_cpu, const float *w_gpu) {
    int fail = 0;
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(w_cpu[i] - w_gpu[i]) / std::abs(w_cpu[i]);
        if (diff > PERCENT_DIFF_ERROR_THRESHOLD / 100.0f) {
        // if( i < 100){
            fail++;
            std::printf("%f ~ %f\n", w_cpu[i], w_gpu[i]);
        }
    }
    std::printf("Number of mismatches: %d\n", fail);
}

void gemver(int n, float alpha, float beta, float *A, float *u1, float *v1, float *u2, float *v2, float *w, float *x, float *y, float *z) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i * n + j] += static_cast<int>(u1[i] * v1[j] + u2[i] * v2[j]);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            x[i] += static_cast<int>(beta * A[j * n + i] * y[j]);
        x[i] += z[i];
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            w[i] += static_cast<int>(alpha * A[i * n + j] * x[j]);
}

__global__ void gemver_kernel1_striped_custom(int n, int row_start, int row_end,
                                              float *a_tile, const float *v1, const float *v2,
                                              const float *u1, const float *u2, int tile_rows) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i_global = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int i_local = (i_global - row_start) % (TILE_BUFFERS * tile_rows);

    if (i_global < row_end && j < n)
        a_tile[i_local * n + j] += u1[i_global] * v1[j] + u2[i_global] * v2[j];
}

__global__ void gemver_kernel2(int n, float beta, const float *a, float *x, const float *y, const float *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int j = 0; j < n; ++j)
            x[i] += static_cast<int>(beta * a[j * n + i] * y[j]);
        x[i] += z[i];
    }
}

__global__ void gemver_kernel3(int n, float alpha, const float *a, const float *x, float *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int j = 0; j < n; ++j)
            w[i] += static_cast<int>(alpha * a[i * n + j] * x[j]);
            // w[j] += static_cast<int>(alpha * a[j * n + i] * x[i]);
    }
}

void gemverCuda(int n, float alpha, float beta,
                float *A, float *u1, float *v1, float *u2, float *v2,
                float *w, float *w_gpu, float *x, float *y, float *z) {

    const int tile_rows = n / NUM_TILES;
    cudaStream_t streams[MAX_STREAMS];

    float *A_gpu, *x_gpu, *y_gpu, *z_gpu, *v1_gpu, *v2_gpu, *u1_gpu, *u2_gpu, *w_gpu_d;

    dim3 block1(THREADS_PER_BLOCK, 1);
    dim3 blockX(THREADS_PER_BLOCK);
    dim3 gridX((n + blockX.x - 1) / blockX.x);

    CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(float) * n * n));
    CHECK_CUDA(cudaMalloc(&x_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&y_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&z_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&w_gpu_d, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&v1_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&v2_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&u1_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc(&u2_gpu, sizeof(float) * n));

    for (int i = 0; i < MAX_STREAMS; ++i)
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

    CHECK_CUDA(cudaMemcpyAsync(y_gpu, y, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(z_gpu, z, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(v1_gpu, v1, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(v2_gpu, v2, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(u1_gpu, u1, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(u2_gpu, u2, sizeof(float) * n, cudaMemcpyHostToDevice, streams[0]));

    CHECK_CUDA(cudaMemsetAsync(x_gpu, 0, sizeof(float) * n, streams[0]));
    CHECK_CUDA(cudaMemsetAsync(w_gpu_d, 0, sizeof(float) * n, streams[0]));

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_TILES; ++t) {
        int stream_id = t % MAX_STREAMS;
        int row_start = t * tile_rows;
        int row_end = (t == NUM_TILES - 1) ? n : (t + 1) * tile_rows;
        int num_rows = row_end - row_start;
        int offset = row_start * n;
        float* tile_base = A_gpu + offset;

        CHECK_CUDA(cudaMemcpyAsync(tile_base, A + offset, sizeof(float) * num_rows * n, cudaMemcpyHostToDevice, streams[stream_id]));

        dim3 grid1((n + block1.x - 1) / block1.x, (num_rows + block1.y - 1) / block1.y);

        gemver_kernel1_striped_custom<<<grid1, block1, 0, streams[stream_id]>>>(
            n, row_start, row_end, tile_base, v1_gpu, v2_gpu, u1_gpu, u2_gpu, tile_rows);
    }

    for (int i = 0; i < MAX_STREAMS; ++i)
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));

    auto end = std::chrono::high_resolution_clock::now();

    gemver_kernel2<<<gridX, blockX>>>(n, beta, A_gpu, x_gpu, y_gpu, z_gpu);
    gemver_kernel3<<<gridX, blockX>>>(n, alpha, A_gpu, x_gpu, w_gpu_d);

    CHECK_CUDA(cudaMemcpy(w_gpu, w_gpu_d, sizeof(float) * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < MAX_STREAMS; ++i)
        CHECK_CUDA(cudaStreamDestroy(streams[i]));

    CHECK_CUDA(cudaFree(A_gpu));
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(y_gpu));
    CHECK_CUDA(cudaFree(z_gpu));
    CHECK_CUDA(cudaFree(w_gpu_d));
    CHECK_CUDA(cudaFree(v1_gpu));
    CHECK_CUDA(cudaFree(v2_gpu));
    CHECK_CUDA(cudaFree(u1_gpu));
    CHECK_CUDA(cudaFree(u2_gpu));

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " ms\n";
}

int main() {
    warmup_cuda_runtime();
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 32 * SHIFTS;

    float alpha, beta;
    float *A;
    CHECK_CUDA(cudaMallocHost(&A, N * N * sizeof(float)));

    float *u1 = (float *)malloc(N * sizeof(float));
    float *v1 = (float *)malloc(N * sizeof(float));
    float *u2 = (float *)malloc(N * sizeof(float));
    float *v2 = (float *)malloc(N * sizeof(float));
    float *w = (float *)malloc(N * sizeof(float));
    float *w_gpu = (float *)malloc(N * sizeof(float));
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *z = (float *)malloc(N * sizeof(float));

    init(N, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    gemverCuda(N, alpha, beta, A, u1, v1, u2, v2, w, w_gpu, x, y, z);
    #ifdef RUN_ON_CPU
        gemver(N, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
        compareResults(N, w, w_gpu);
    #else
        // for (int i = 0; i < N; ++i)
        //     std::cerr << w_gpu[i] << " ";
        // std::cerr << std::endl;
    #endif
    CHECK_CUDA(cudaFreeHost(A));
    free(u1); free(v1); free(u2); free(v2);
    free(w); free(w_gpu); free(x); free(y); free(z);
    auto end = std::chrono::high_resolution_clock::now();
    #ifndef RUN_ON_CPU
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Total Execution Time: " << elapsed.count() << " ms\n";
    #endif

    return 0;
}
