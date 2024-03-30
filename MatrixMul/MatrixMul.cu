#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>

#define N 1024

// gpu kernel callable from device and host
__global__ 
void mult_gpu(int* d_a_in, int* d_b_in, int* d_c_out, int NN)
{
    // compute the 1d thread id based on the block and thread ids
    int row= blockIdx.x * blockDim.x + threadIdx.x;
    int col= blockIdx.y * blockDim.y + threadIdx.y;

    int tmp = 0 ; 
    // do a bound check and return if out of bounds
    for( int k = 0 ; k < NN ; k++ )
    {
        tmp += d_a_in[row*NN + k] * d_b_in[k*NN + col];
    }

    // add the two input arrays and store the result in the output array
     d_c_out[row*NN + col] = tmp; 
};

void mult_cpu(int* h_a_in, int* h_b_in, int* h_c_out, int NN)
{
    for(int i=0; i<NN; i++)
    {
        for(int j=0; j<NN; j++)
        {
            for(int k=0; k<NN; k++)
            {
                h_c_out[i*NN+j] += h_a_in[i*NN+k] * h_b_in[k*NN+j];
            }
        }
    }
};

int main()
{
    // create device pointers 
    std::vector<int> h_a_in(N*N);
    std::vector<int> h_b_in(N*N); 
    std::vector<int> h_c_out(N*N); 
    std::vector<int> h_c_out_gpu(N*N); 

    // iniatilize the matrices
    std::iota(h_a_in.begin(), h_a_in.end(), 0);
    std::iota(h_b_in.begin(), h_b_in.end(), 0);
	
    // call the CPU routine
    auto start_cpu = std::chrono::high_resolution_clock::now();
	mult_cpu(h_a_in.data(), h_b_in.data(), h_c_out.data(), N); 
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << diff_cpu.count() << " s\n";


    //allocate device memory
    int *d_a_in, *d_b_in, *d_c_out;  
    cudaMalloc(&d_a_in, N*N*sizeof(int));
    cudaMalloc(&d_b_in, N*N*sizeof(int));
    cudaMalloc(&d_c_out, N*N*sizeof(int));

    //copy the pointer to the device
    cudaMemcpy(d_a_in, h_a_in.data(), N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b_in.data(),N*N*sizeof(int), cudaMemcpyHostToDevice);
    
    // assume N is divisible by 32
    static_assert( N % 32 == 0, "N must be divisible by 32");
    dim3 block(32,32,1);
    dim3 grid(N/32,N/32,1);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    // call the gpu kernel 
    mult_gpu<<<grid,block>>>(d_a_in, d_b_in, d_c_out, N);
    // wait for gpu to finish
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
    std::cout << "GPU time: " << diff_gpu.count() << " s\n";

    cudaMemcpy(h_c_out_gpu.data(), d_c_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    // check they are equal 
    if( std::equal(h_c_out.data(), h_c_out.data()+N*N, h_c_out_gpu.data()) ) printf("Success\n");
    else printf("Error\n");

    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }

    // free memeory of the device and host
    cudaFree(d_a_in);
    cudaFree(d_b_in);
    cudaFree(d_c_out);


    return 0;
}