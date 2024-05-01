#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>

struct Params
{
    float eps;
    float sigma;
    float dt;
    float kb;
};

__constant__ Params MyParams;

#define cudaCheckErrors(call)                                             \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(1);                                                     \
        }                                                                \
    }

#define N 1024
#define THREADS_PER_BLOCK 32
#define TILE_SIZE THREADS_PER_BLOCK
#define BOX_SIZE 60.0f

__device__ float3
tile_compute_acceleration(float4 myPos, float4 *sharedPos, float3 acc)
{
    float eps = MyParams.eps;
    float sigma = MyParams.sigma;
    float sigma6 = powf(sigma, 6);
    float3 r;
	float dist;
    for (int i = 0; i < TILE_SIZE; i+= 4 )
    {
        r.x = sharedPos[i].x - myPos.x;
        r.y = sharedPos[i].y - myPos.y;
        r.z = sharedPos[i].z - myPos.z;

        dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (dist > 1e-6f)
        {
            float rInv = 1.0f / dist;
            float rInv6 = powf(rInv, 6);
            float rInv12 = rInv6 * rInv6;
            float forceScalar = 24.0f * eps * (2 * sigma6 * sigma6 * rInv12 - sigma6 * rInv6) * rInv;

            acc.x += r.x * forceScalar;
            acc.y += r.y * forceScalar;
            acc.z += r.z * forceScalar;
        }

        r.x = sharedPos[i+1].x - myPos.x;
        r.y = sharedPos[i+1].y - myPos.y;
        r.z = sharedPos[i+1].z - myPos.z;

        dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (dist > 1e-6f)
        {
            float rInv = 1.0f / dist;
            float rInv6 = powf(rInv, 6);
            float rInv12 = rInv6 * rInv6;
            float forceScalar = 24.0f * eps * (2 * sigma6 * sigma6 * rInv12 - sigma6 * rInv6) * rInv;

            acc.x += r.x * forceScalar;
            acc.y += r.y * forceScalar;
            acc.z += r.z * forceScalar;
        }

        r.x = sharedPos[i+2].x - myPos.x;
        r.y = sharedPos[i+2].y - myPos.y;
        r.z = sharedPos[i+2].z - myPos.z;

        dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (dist > 1e-6f)
        {
            float rInv = 1.0f / dist;
            float rInv6 = powf(rInv, 6);
            float rInv12 = rInv6 * rInv6;
            float forceScalar = 24.0f * eps * (2 * sigma6 * sigma6 * rInv12 - sigma6 * rInv6) * rInv;

            acc.x += r.x * forceScalar;
            acc.y += r.y * forceScalar;
            acc.z += r.z * forceScalar;
        }

        r.x = sharedPos[i+3].x - myPos.x;
        r.y = sharedPos[i+3].y - myPos.y;
        r.z = sharedPos[i+3].z - myPos.z;

        dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (dist > 1e-6f)
        {
            float rInv = 1.0f / dist;
            float rInv6 = powf(rInv, 6);
            float rInv12 = rInv6 * rInv6;
            float forceScalar = 24.0f * eps * (2 * sigma6 * sigma6 * rInv12 - sigma6 * rInv6) * rInv;

            acc.x += r.x * forceScalar;
            acc.y += r.y * forceScalar;
            acc.z += r.z * forceScalar;
        }
    }

    return acc;
}

__global__ void
compute_acceleration(float4 *__restrict pos, float4 *__restrict acc)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    __shared__ float4 sharedPos[TILE_SIZE];

    float3 acceleration = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < N; i += TILE_SIZE)
    {
        sharedPos[tid] = pos[i + tid];
        __syncthreads();

        acceleration = tile_compute_acceleration(pos[gtid], sharedPos, acceleration);
        __syncthreads();
    }
    acc[gtid] = make_float4(acceleration.x, acceleration.y, acceleration.z, 0.0f);
}

__global__ void
leapfrog_integration(float4 *pos, float4 *vel, float4 *acc, float4 *acc_old)
{
    const float dt = MyParams.dt;
    const float box_size = BOX_SIZE;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
	float4 velocity = vel[gtid]; 
	float4 accelration = acc[gtid]; 
	float4 accelration_old = acc_old[gtid]; 
    
    pos[gtid].x += velocity.x * dt + 0.5f * accelration.x * dt * dt;
    pos[gtid].y += velocity.y * dt + 0.5f * accelration.y * dt * dt;
    pos[gtid].z += velocity.z * dt + 0.5f * accelration.z * dt * dt;

    vel[gtid].x += 0.5f * (accelration_old.x + accelration.x) * dt;
    vel[gtid].y += 0.5f * (accelration_old.y + accelration.y) * dt;
    vel[gtid].z += 0.5f * (accelration_old.z + accelration.z) * dt;

    if (pos[gtid].x < 0.0f || pos[gtid].x > box_size)
    {
        pos[gtid].x = pos[gtid].x < 0.0f ? 0.0 : box_size;
        vel[gtid].x = -vel[gtid].x;
    }
    if (pos[gtid].y < 0.0f || pos[gtid].y > box_size)
    {
        pos[gtid].y = pos[gtid].y < 0.0f ? 0.0 : box_size;
        vel[gtid].y = -vel[gtid].y;
    }
    if (pos[gtid].z < 0.0f || pos[gtid].z > box_size)
    {
        pos[gtid].z = pos[gtid].z < 0.0f ? 0.0 : box_size;
        vel[gtid].z = -vel[gtid].z;
    }

    acc_old[gtid] = acc[gtid];
}

__global__ void
rescale(float4 *vel, float T_Ref, float KE)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sdata[1];
    sdata[0] = 0.0f;
    for(int i = 0; i < N; i++)
    {
        sdata[0] += vel[gtid].x * vel[gtid].x + vel[gtid].y * vel[gtid].y + vel[gtid].z * vel[gtid].z;
    }
    __syncthreads();

    atomicAdd(&KE, sdata[0]);

    float kb= MyParams.kb;
    float avKE = KE / N;
    float Tnow = 2.0f * avKE / 3.0f / kb;
    float lam = sqrtf(T_Ref / Tnow);

    vel[gtid].x = vel[gtid].x * lam;
	
	KE = lam; 
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(float4 *g_idata, const unsigned int n, float* result  )
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n)
    {
		float4 g_idata_i = g_idata[i]; 
		float4 g_data_i_blocks_size =  g_idata[i + blockSize]; 
        sdata[tid] += g_idata_i.x*g_idata_i.x + g_idata_i.y*g_idata_i.y + g_idata_i.z*g_idata_i.z + g_data_i_blocks_size.x+g_data_i_blocks_size.x + g_data_i_blocks_size.y+g_data_i_blocks_size.y + g_data_i_blocks_size.z+g_data_i_blocks_size.z;
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
	{
        atomicAdd(result, sdata[0]);
	}
}

__global__ void
rescale_v2(float4 *vel, float T_Ref, float* KE)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    float kb= MyParams.kb;
    float avKE = *KE / N;
    float Tnow = 2.0f * avKE / 3.0f / kb;
    float lam = sqrtf(T_Ref / Tnow);
    
    vel[gtid].x = vel[gtid].x * lam;
	
	*KE = lam; 
}

void write_poistion_to_file(float4 *pos, int time_step);

int main(int argc, char *argv[])
{
    float4 *h_pos = new float4[N];
    float4 *h_vel = new float4[N];
    float4 *h_acc = new float4[N];
    float4 *h_acc_old = new float4[N];

    float box_size = BOX_SIZE;
    int TimeSteps = 50;

    Params hostParams = {0.1f, 1.6f, 0.0001f, 0.83144f};
    cudaMemcpyToSymbol(MyParams, &hostParams, sizeof(Params));
    
    srand(time(0));
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        h_pos[i].x = static_cast<float>(rand()) / RAND_MAX * box_size;
        h_pos[i].y = static_cast<float>(rand()) / RAND_MAX * box_size;
        h_pos[i].z = static_cast<float>(rand()) / RAND_MAX * box_size;
        h_pos[i].w = static_cast<float>(rand()) / RAND_MAX * box_size;

        h_vel[i].x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_vel[i].y = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_vel[i].z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_vel[i].w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;

        h_acc_old[i].x = 0.0f;
        h_acc_old[i].y = 0.0f;
        h_acc_old[i].z = 0.0f;
        h_acc_old[i].w = 0.0f;
    }

    float4 *d_pos, *d_vel, *d_acc, *d_acc_old;
    cudaCheckErrors(cudaMalloc((void **)&d_pos, N * sizeof(float4)));
    cudaCheckErrors(cudaMalloc((void **)&d_vel, N * sizeof(float4)));
    cudaCheckErrors(cudaMalloc((void **)&d_acc, N * sizeof(float4)));
    cudaCheckErrors(cudaMalloc((void **)&d_acc_old, N * sizeof(float4)));

    for (int i = 0; i < TimeSteps; i++)
    {
        write_poistion_to_file(h_pos, i);

        cudaCheckErrors(cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice));
        cudaCheckErrors(cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice));
        cudaCheckErrors(cudaMemcpy(d_acc_old, h_acc_old, N * sizeof(float4), cudaMemcpyHostToDevice));

        compute_acceleration<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_pos, d_acc);

        float KE = 0.0f;
        float *d_KE;
        cudaCheckErrors(cudaMalloc(&d_KE, sizeof(float)));
        cudaCheckErrors(cudaMemcpy(d_KE, &KE, sizeof(float), cudaMemcpyHostToDevice));

        reduce6<THREADS_PER_BLOCK><<<N / 2 /  THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>(d_vel, 2, d_KE);

        rescale_v2<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_vel, 100000.0f, d_KE);

        leapfrog_integration<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_pos, d_vel, d_acc, d_acc_old);

        cudaCheckErrors(cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost));
        cudaCheckErrors(cudaDeviceSynchronize());
    }
}

void write_poistion_to_file(float4 *pos, int timestep)
{
    std::string filename = "output_" + std::to_string(timestep) + ".txt";
    std::ofstream file(filename);

    file << "ITEM: TIMESTEP\n";
    file << timestep << "\n";
    file << "ITEM: NUMBER OF ATOMS\n";
    file << N << "\n";
    file << "ITEM: BOX BOUNDS ff ff ff\n";
    file << "0 " << BOX_SIZE << "\n";
    file << "0 " << BOX_SIZE << "\n";
    file << "0 " << BOX_SIZE << "\n";
    file << "ITEM: ATOMS id mol type x y z\n";

    for (int i = 0; i < N; i++)
    {
		int color = i%2 ; 
        file << i << " " << 0 << " " << color << " "
           << pos[i].x << " " << pos[i].y << " " << pos[i].z << "\n";
    }
}
