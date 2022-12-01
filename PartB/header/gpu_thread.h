// Create other necessary functions here
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end)                                            \
    std::chrono::duration_cast<gran>(end - start).count()

__global__ void matrixMul_ref(const int *matA, const int *matB, int *output,
                              int N) {

    // Compute each thread's global row and column index

    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int rowA = rowC * 2;
    int colB = colC * 2;
    // Iterate over row, and down column
    // c[row * N + col] = -1;
    int sum = 0;
    for (int iter = 0; iter < N; iter++) {
        sum += matA[rowA * N + iter] * matB[iter * N + colB];
        sum += matA[(rowA + 1) * N + iter] * matB[iter * N + colB];
        sum += matA[rowA * N + iter] * matB[iter * N + (colB + 1)];
        sum += matA[(rowA + 1) * N + iter] * matB[iter * N + (colB + 1)];
        // Accumulate results for a single element
    }
    int indexC = rowC * (N >> 1) + colC;
    output[indexC] = sum;
}

__global__ void matrixMul1(const int *matA, const int *matB, int *output,
                           int N) {

    // Compute each thread's global row and column index

    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int rowA = rowC * 2;
    int colB = colC * 2;
    // Iterate over row, and down column
    // c[row * N + col] = -1;
    int sum = 0;
    for (int iter = 0; iter < N; iter++) {
        sum += (matA[rowA * N + iter] + matA[(rowA + 1) * N + iter]) *
               (matB[iter * N + colB] + matB[iter * N + (colB + 1)]);
    }
    int indexC = rowC * (N >> 1) + colC;
    output[indexC] = sum;
}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output) {
    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(int);

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes / 4);

    // Copy data to the device
    cudaMemcpy(d_a, matA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, matB, bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / (2 * THREADS);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    auto begin = TIME_NOW;
    matrixMul1<<<blocks, threads>>>(d_a, d_b, d_c, N);
    auto end = TIME_NOW;
    cout << "Cuda reference execution time: "
         << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0
         << " ms\n";

    // Copy back to the host
    cudaMemcpy(output, d_c, bytes / 4, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return;
}
