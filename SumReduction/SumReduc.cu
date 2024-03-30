#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>
#include <iostream>

#define N 4096*4096
#define BLOCK_SIZE 256

// gpu kernel callable from device and host
__global__ 
void sum_gpu(int* d_in, int* d_out)
{
    // declare a shared memeory with the block size
    __shared__ int sdata[BLOCK_SIZE];

    // read in the memory from the shared data, since we are using half the grid we have to have a stride of blockdimx
    int index = blockIdx.x * (blockDim.x *2)  + threadIdx.x;
    sdata[threadIdx.x] = d_in[index] + d_in[index + blockDim.x]; 
    __syncthreads();

    // loop over the and divide the stride by 2 for each haldf
    for( int s = blockDim.x/2; s>32; s>>=1)
    {
        if(threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // unroll the last loop 
    if( threadIdx.x < 32 )
    {
        sdata[threadIdx.x] += sdata[threadIdx.x + 32];
        sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        sdata[threadIdx.x] += sdata[threadIdx.x + 1];
    }

    // write thew result 
    if(threadIdx.x == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
};

int main()
{
    // create device pointers 
    int* h_in = (int*)malloc(N*sizeof(int));
    int *h_out = (int*)malloc(N*sizeof(int));
	int* h_out_gpu = (int*)malloc(N*sizeof(int));

    // fill the input arrays with some data
    for(int i=0; i<N; i++)
    {
        h_in[i] = i;

    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    h_out[0] = std::accumulate(h_in, h_in+N, 0);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << diff_cpu.count() << " s\n";



    //allocate device memory
    int *d_in, *d_out; 
    cudaMalloc(&d_in, N*sizeof(int));
    cudaMalloc(&d_out, N*sizeof(int));

    //copy the pointer to the device
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);


    auto start_gpu = std::chrono::high_resolution_clock::now();
    // call the gpu kernel 
    sum_gpu<<<N/BLOCK_SIZE/2,BLOCK_SIZE>>>(d_in, d_out);

    sum_gpu<<<1,BLOCK_SIZE>>>(d_out, d_out);
    // wait for gpu to finish
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
    std::cout << "GPU time: " << diff_gpu.count() << " s\n";
	
	// copy the memeory back to cpu to compare
	cudaMemcpy(h_out_gpu, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
		
    // check they are equal 
    if( h_out_gpu[0] == h_out[0] ) printf("Success\n");
    else printf("Error\n");

    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }

    // free memeory of the device and host
    free(h_in);
    free(h_out);
	free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);


    return 0;
}