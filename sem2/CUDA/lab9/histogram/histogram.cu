#include "histogram.h"
#include <iostream>

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;

    int binWidth = ceil (26.0/ nBins ) ;

    if (x < size)
    {
        int alphabetPos = buffer[x] - 'a';

        if (alphabetPos >= 0 && alphabetPos < 26)
        {
            atomicAdd(&histogram[alphabetPos/binWidth], 1);
        }
    }
}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    int binWidth = ceil(26.0/nBins);

    for (long i = tid; i < size; i += stride)
    {
        int alphabetPos = buffer[i] - 'a';
        if (alphabetPos >= 0 && alphabetPos < 26)
            atomicAdd(&histogram[alphabetPos / binWidth],1);
    }
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    extern __shared__ unsigned int localHist[];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;

    int binWidth = ceil(26.0/nBins);

    for (int i = 0; i < nBins; i++)
        localHist[i] = 0;
    __syncthreads();

    for (long i = blockStart+tid; i<size; i = i + stride)
    {
        int alphabetPos = buffer[i] - 'a';
        if (alphabetPos >= 0 && alphabetPos < 26)
            atomicAdd(&localHist[alphabetPos/binWidth], 1);
    }
    __syncthreads();

    for (int i = tid; i<nBins; i += blockDim.x)
        atomicAdd(&histogram[i], localHist[i]);
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    unsigned char* d_data;
    unsigned int* d_out_hist;

    cudaMalloc((void**)&d_data, data.size() * sizeof(unsigned char));
    cudaMalloc((void**)&d_out_hist, nBins * sizeof(unsigned int));

    cudaMemset(d_out_hist, 0, nBins * sizeof(unsigned int));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (data.size() + threadsPerBlock - 1) / threadsPerBlock;

    histogram_3<<<blocks, threadsPerBlock, nBins*sizeof(unsigned int)>>>(d_data, data.size(), d_out_hist, nBins);
    cudaDeviceSynchronize();

    std::vector<unsigned int> h_histogram(nBins);
    cudaMemcpy(h_histogram.data(), d_out_hist, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_out_hist);

    cudaDeviceReset();
    cudaDeviceSynchronize();

    return h_histogram;
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// Funkcja wielostrumieniowa bez cudaMallocHost
std::vector<unsigned int> computeHistogramOnDeviceStreams(
    const std::vector<unsigned char> &data, int nBins, int nStreams)
{
    size_t dataSize = data.size();
    size_t streamSize = dataSize / nStreams;
    size_t remainder = dataSize % nStreams;

    unsigned int* d_histograms;
    checkCudaErrors(cudaMalloc((void**)&d_histograms, nStreams * nBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_histograms, 0, nStreams * nBins * sizeof(unsigned int)));

    std::vector<unsigned int> h_histograms(nStreams * nBins, 0);

    std::vector<cudaStream_t> streams(nStreams);
    for (int i = 0; i < nStreams; i++)
        checkCudaErrors(cudaStreamCreate(&streams[i]));

    int threadsPerBlock = 256; // zmniejszone dla bezpieczeństwa
    int blocksPerGrid = 256;

    std::vector<unsigned char*> d_dataArray(nStreams, nullptr);

    for (int i = 0; i < nStreams; i++)
    {
        size_t currentStreamSize = streamSize;
        if (i == nStreams - 1)
            currentStreamSize += remainder;

        checkCudaErrors(cudaMalloc(&d_dataArray[i], currentStreamSize * sizeof(unsigned char)));

        checkCudaErrors(cudaMemcpyAsync(d_dataArray[i],
                                        data.data() + i * streamSize,
                                        currentStreamSize * sizeof(unsigned char),
                                        cudaMemcpyHostToDevice,
                                        streams[i]));

        histogram_3<<<blocksPerGrid, threadsPerBlock, nBins * sizeof(unsigned int), streams[i]>>>(
            d_dataArray[i], currentStreamSize, d_histograms + i * nBins, nBins
        );
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpyAsync(h_histograms.data() + i * nBins,
                                        d_histograms + i * nBins,
                                        nBins * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost,
                                        streams[i]));
    }

    for (int i = 0; i < nStreams; i++)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaFree(d_dataArray[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaFree(d_histograms));

    std::vector<unsigned int> finalHistogram(nBins, 0);
    for (int i = 0; i < nStreams; i++)
        for (int j = 0; j < nBins; j++)
            finalHistogram[j] += h_histograms[i * nBins + j];

    return finalHistogram;
}



std::vector<unsigned int> computeHistogramOnHost(const std::vector<unsigned char> &data, int nBins)
{
    std::vector<unsigned int> histogram(nBins, 0);
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division

    for (const auto &ch : data)
    {
        int alphabetPosition = ch - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
        {
            histogram[alphabetPosition / binWidth]++;
        }
    }

    return histogram;
}
