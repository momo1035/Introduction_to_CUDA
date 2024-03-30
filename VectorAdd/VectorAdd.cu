#include <stdlib.h>
#include <algorithm>
#include <functional>

#define N 1000

// gpu kernel callable from device and host
__global__ 
void add_gpu(int* d_a_in, int* d_b_in, int* d_c_out, int NN)
{
    // compute the 1d thread id based on the block and thread ids
    int tid= blockIdx.x * blockDim.x + threadIdx.x;

    // do a bound check and return if out of bounds
    if ( tid < NN )
    {
        // add the two input arrays and store the result in the output array
        d_c_out[tid] = d_a_in[tid] + d_b_in[tid];
    }
};

int main()
{
    // create device pointers 
    int* h_a_in = (int*)malloc(N*sizeof(int));
    int *h_b_in = (int*)malloc(N*sizeof(int));
    int *h_c_out = (int*)malloc(N*sizeof(int));
	int* h_c_out_gpu = (int*)malloc(N*sizeof(int));


    // fill the input arrays with some data
    for(int i=0; i<N; i++)
    {
        h_a_in[i] = i;
        h_b_in[i] = i+1;
    }

    //allocate device memory
    int *d_a_in, *d_b_in, *d_c_out; 
    cudaMalloc(&d_a_in, N*sizeof(int));
    cudaMalloc(&d_b_in, N*sizeof(int));
    cudaMalloc(&d_c_out, N*sizeof(int));

    //copy the pointer to the device
    cudaMemcpy(d_a_in, h_a_in, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b_in, N*sizeof(int), cudaMemcpyHostToDevice);

    //compute the sum with std libray
    std::transform(h_a_in, h_a_in+N, h_b_in, h_c_out, std::plus<int>());

    // call the gpu kernel 
    add_gpu<<<1,N>>>(d_a_in, d_b_in, d_c_out, N);
    // wait for gpu to finish
    cudaDeviceSynchronize();
	
	// copy the memeory back to cpu to compare
	cudaMemcpy(h_c_out_gpu, d_c_out, N*sizeof(int), cudaMemcpyDeviceToHost);
		
    // check they are equal 
    if( std::equal(h_c_out, h_c_out+N, h_c_out_gpu) ) printf("Success\n");
    else printf("Error\n");

    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }

    // free memeory of the device and host
    free(h_a_in);
    free(h_b_in);
    free(h_c_out);
	free(h_c_out_gpu);
    cudaFree(d_a_in);
    cudaFree(d_b_in);
    cudaFree(d_c_out);


    return 0;
}