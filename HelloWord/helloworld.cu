#include <stdlib.h>
#include <stdio.h>
// gpu kernel callable from device and host
__global__ 
void gpuhelloword(void)
{
    printf("Hello World from GPU\n");
};
int main()
{
    // call gpu kernel
    gpuhelloword<<<1,16>>>();
    // wait for gpu to finish
    cudaDeviceSynchronize();
    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
    }
    return 0;
}