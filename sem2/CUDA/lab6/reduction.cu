#include "reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void reductionKernelBasic(int *sum, int *input, int width)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < width) ? input[idx] : 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (tid % (2*stride) == 0 && tid + stride < blockDim.x)
            sdata[tid] += sdata[tid + stride];
    }

    if (tid == 0)
        atomicAdd(sum, sdata[0]);
}



__global__ void reductionKernelOptimized(int *blockSums, int *input, int width)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < width) ? input[idx] : 0;
    __syncthreads();

    int n = blockDim.x;

    for (int stride = n / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        blockSums[blockIdx.x] = sdata[0];
}

__global__ void reductionKernelCooperativeGroups(int *sum, const int *input, int width)
{
    
}


int reductionOnDevice(const std::vector<int> &data, ReductionMethod method)
{
    int *d_input, *d_sum;
    size_t n = data.size();

    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMemcpy(d_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = (n < 1024) ? n : 1024;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    int output = 0;

    if (method == ReductionMethod::Basic)
    {
        cudaMalloc((void**)&d_sum, sizeof(int));
        cudaMemset(d_sum, 0, sizeof(int));
        reductionKernelBasic<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_sum, d_input, n);
        cudaMemcpy(&output, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_sum);
    }
    else if (method == ReductionMethod::Optimized)
    {
        // Alokacja tablicy na sumy bloków
        cudaMalloc((void**)&d_sum, blocks * sizeof(int));

        reductionKernelOptimized<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_sum, d_input, n);

        // Kopiujemy wyniki bloków na hosta
        std::vector<int> h_blockSums(blocks);
        cudaMemcpy(h_blockSums.data(), d_sum, blocks * sizeof(int), cudaMemcpyDeviceToHost);

        // Sumujemy wyniki bloków na hosta
        output = 0;
        for (auto val : h_blockSums)
            output += val;

        cudaFree(d_sum);
    }

    cudaFree(d_input);
    return output;
}

int reductionOnHost(const std::vector<int> &data)
{
    int sum = 0;
    for (const auto &val : data)
    {
        sum += val;
    }
    return sum;
}
