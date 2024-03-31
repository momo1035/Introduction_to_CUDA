#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>
#include "GenMat.cpp"

#define N 8192
#define BLOCK_SIZE 1024

// gpu kernel callable from device and host
__global__ void spmv(int *d_in, int *d_out, int NN, int *d_ptr, int *d_values, int *d_colind)
{

    int row = blockDim.x *  blockIdx.x + threadIdx.x;
    if (row < NN)
    {
        float tmp = 0;
        int row_start = d_ptr[row];
        int row_end = d_ptr[row + 1];
        for (int j = row_start; j < row_end; j++)
            tmp += d_values[j] * d_in[d_colind[j]];
        d_out[row] += tmp;
    }
};

void mult_cpu(int *h_in, int *h_out, int NN, int *h_ptr, int *h_values, int *h_colind)
{
    // loop over the rows
    for (int i = 0; i < NN; i++)
    {
        // loop over the columns
        int startrow = h_ptr[i];
        int endtrow = h_ptr[i + 1];

        int tmp = 0;
        // perform a dot product between the row and colums
        for (int j = startrow; j < endtrow; j++)
        {
            tmp += h_values[j] * h_in[h_colind[j]];
        }

        h_out[i] = tmp;
    }
};

int main()
{
    // create device pointers
    int *h_in = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    int *h_out_gpu = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        h_in[i] = 1;
    }

    // fill the input arrays with some data
    std::vector<int> data, indices, ptr;
    generate_matrix_data(data, indices, ptr, N*N);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    mult_cpu(h_in, h_out, N, ptr.data(), data.data(), indices.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << diff_cpu.count() << " s\n";

    // allocate device memory
    int *d_in, *d_out, *d_ptr, *d_values, *d_colind;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_ptr, N * N * sizeof(int));
    cudaMalloc(&d_values, N * N * sizeof(int));
    cudaMalloc(&d_colind, N * N * sizeof(int));

    // copy the pointer to the device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, ptr.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, data.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colind, indices.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    // call the gpu kernel
    spmv<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, N, d_ptr, d_values, d_colind);
    // wait for gpu to finish
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
    std::cout << "GPU time: " << diff_gpu.count() << " s\n";

    // copy the memeory back to cpu to compare
    cudaMemcpy(h_out_gpu, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }


    // check they are equal
    if (std::equal(h_out, h_out + N, h_out_gpu))
        printf("Success\n");
    else
        printf("Error\n");

    // free memeory of the device and host
    free(h_in);
    free(h_out);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_ptr);
    cudaFree(d_values);
    cudaFree(d_colind);

    return 0;
}