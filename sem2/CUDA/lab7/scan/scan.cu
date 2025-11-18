#include "scan.h"


__global__ void kernelScan(int *out, const int *in, size_t n)
{
    extern __shared__ int sdata[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_idx < n)
        sdata[threadIdx.x] = in[global_idx];
    else
        sdata[threadIdx.x] = 0;

    __syncthreads();

    for(int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int val = 0;
        if(threadIdx.x >= offset)
            val = sdata[threadIdx.x - offset];
        __syncthreads();
        sdata[threadIdx.x] += val;
        __syncthreads();
    }

    if(global_idx < n)
        out[global_idx] = sdata[threadIdx.x];
}


__global__ void kernelExtractSums(const int *d_out, int *d_blockSums, int n, int blockSize) {
    int blockId = blockIdx.x;
    int lastIdx = min((blockId + 1) * blockSize, n) - 1;
    d_blockSums[blockId] = d_out[lastIdx];
}



__global__ void kernelAddSums(int *out, const int *sums, size_t n)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_idx >= n) return;      
    if(blockIdx.x == 0) return;        

    out[global_idx] += sums[blockIdx.x - 1];
}



std::vector<int> computeBlockSumsCPU(const std::vector<int> &gpuScan, int blockSize)
{
    size_t n = gpuScan.size();
    int numBlocks = (n + blockSize - 1) / blockSize;
    std::vector<int> blockSums(numBlocks);

    for(int b = 0; b < numBlocks; ++b) {
        int lastIdx = std::min((b + 1) * blockSize, (int)n) - 1;
        blockSums[b] = gpuScan[lastIdx];
    }

    return blockSums;
}

std::vector<int> scanOnCPU(const std::vector<int> &in)
{
    std::vector<int> out(in.size());
    if(in.empty()) return out;
    out[0] = in[0];
    for(size_t i = 1; i < in.size(); ++i)
        out[i] = out[i-1] + in[i];
    return out;
}

// =================================================================================================================================
// ===============================================================Kod do liczneia na GPU============================================
// =================================================================================================================================

std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
{
    size_t n = in.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);
    int numBlocks = gridSize;

    int *d_in, *d_out, *d_blockSums;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    if (method == ScanMethod::KoggeStone) {
        kernelScan<<<gridSize, blockSize, sharedMemSize>>>(d_out, d_in, n);
        cudaDeviceSynchronize(); // upewniamy się, że kernel skończył

        // kopiowanie wyników i print
        std::vector<int> out(n);
        cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "After kernelScan:" << std::endl;
        // for (size_t i = 0; i < n; ++i) std::cout << out[i] << " ";
        // std::cout << std::endl;

        kernelExtractSums<<<numBlocks, blockSize>>>(d_out, d_blockSums, n, 512);

        cudaDeviceSynchronize();

        std::vector<int> blockSums(numBlocks);
        cudaMemcpy(blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "After kernelExtractSums:" << std::endl;
        // for (int val : blockSums) std::cout << val << " ";
        // std::cout << std::endl;

        kernelScan<<<1, numBlocks, numBlocks * sizeof(int)>>>(d_blockSums, d_blockSums, numBlocks);
        cudaDeviceSynchronize();

        // cudaMemcpy(blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "After scan of blockSums:" << std::endl;
        // for (int val : blockSums) std::cout << val << " ";
        // std::cout << std::endl;

        kernelAddSums<<<gridSize, blockSize>>>(d_out, d_blockSums, n);
        cudaDeviceSynchronize();

        cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "After kernelAddSums:" << std::endl;
        // for (size_t i = 0; i < n; ++i) std::cout << out[i] << " ";
        // std::cout << std::endl;

        return out;
    }


    cudaDeviceSynchronize();

    std::vector<int> out(n);
    cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);

    return out;
}


// =================================================================================================================================
// ===============================================================Kod do liczneia na CPU============================================
// =================================================================================================================================

// std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
// {
//     size_t n = in.size();
//     int blockSize = 128;
//     int gridSize = (n + blockSize - 1) / blockSize;

//     int *d_in, *d_out;
//     cudaMalloc(&d_in, n * sizeof(int));
//     cudaMalloc(&d_out, n * sizeof(int));
//     cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

//     kernelScan<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_out, d_in, n);
//     cudaDeviceSynchronize();

//     std::vector<int> gpuScan(n);
//     cudaMemcpy(gpuScan.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

//     std::vector<int> blockSums = computeBlockSumsCPU(gpuScan, blockSize);

//     blockSums = scanOnCPU(blockSums);

//     int *d_blockSums;
//     cudaMalloc(&d_blockSums, blockSums.size() * sizeof(int));
//     cudaMemcpy(d_blockSums, blockSums.data(), blockSums.size() * sizeof(int), cudaMemcpyHostToDevice);

//     kernelAddSums<<<gridSize, blockSize>>>(d_out, d_blockSums, n);
//     cudaDeviceSynchronize();

//     cudaMemcpy(gpuScan.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

//     cudaFree(d_in);
//     cudaFree(d_out);
//     cudaFree(d_blockSums);

//     return gpuScan;
// }

std::vector<int> scanOnHost(const std::vector<int> &in)
{
    std::vector<int> out(in.size());
    if (in.size() == 0)
    {
        return out;
    }

    out[0] = in[0];
    for (size_t i = 1; i < in.size(); ++i)
    {
        out[i] = out[i - 1] + in[i];
    }

    return out;
}
