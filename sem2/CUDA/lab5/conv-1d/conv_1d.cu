#include "conv_1d.h"

__constant__ float c_mask[MAX_MASK_WIDTH];

__global__ void conv1dBasicKernel(float *output, const float *signal, const float *mask, const int width, const int maskWidth)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < width)
    {
        float sum = 0.0f;
        int halfMask = maskWidth / 2;

        for (int i = 0; i < maskWidth; i++)
        {
            int signalIdx = x + i - halfMask;
            if (signalIdx >= 0 && signalIdx < width)
            {
                sum += signal[signalIdx] * mask[i];
            }
        }

        output[x] = sum;
    }
}


__global__ void conv1dTiledKernel(float *output, const float *signal, int width, int maskWidth)
{
    extern __shared__ float s_signal[];  // rozmiar = blockDim.x + maskWidth - 1

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    int radius = maskWidth / 2;
    int start = bx * bdx;
    int x = start + tx;

    int sharedIdx = tx + radius;

    // Wczytanie środka sygnału
    if (x < width)
        s_signal[sharedIdx] = signal[x];
    else
        s_signal[sharedIdx] = 0.0f;

    // Wczytanie lewej krawędzi
    if (tx < radius)
    {
        int leftIdx = start + tx - radius;
        s_signal[tx] = (leftIdx >= 0) ? signal[leftIdx] : 0.0f;
    }

    // Wczytanie prawej krawędzi
    if (tx >= bdx - radius)
    {
        int rightIdx = start + tx + radius;
        int sharedRightIdx = sharedIdx + radius;
        if (sharedRightIdx < bdx + maskWidth - 1)  // zabezpieczenie przed out-of-bounds
            s_signal[sharedRightIdx] = (rightIdx < width) ? signal[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Wykonanie konwolucji
    if (x < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < maskWidth; ++k)
            sum += s_signal[tx + k] * c_mask[k];

        output[x] = sum;
    }
}


std::vector<float> convolutionOnDevice(const std::vector<float> &signal, const std::vector<float> &mask, ConvMethod method)
{
    float *d_signal = nullptr;
    float *d_mask = nullptr;
    float *d_output = nullptr;

    std::vector<float> convSignal(signal.size());

    cudaMalloc((void**)&d_signal, signal.size() * sizeof(float));
    cudaMalloc((void**)&d_output, signal.size() * sizeof(float));
    cudaMalloc((void**)&d_mask, mask.size() * sizeof(float));

    cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (signal.size() + threadsPerBlock - 1) / threadsPerBlock;

    if (method == ConvMethod::Basic)
    {
        conv1dBasicKernel<<<blocks, threadsPerBlock>>>(d_output, d_signal, d_mask, signal.size(), mask.size());
    }
    else if (method == ConvMethod::Tiled)
    {
        cudaMemcpyToSymbol(c_mask, mask.data(), mask.size() * sizeof(float));

        size_t sharedMemSize = (threadsPerBlock + 2 * mask.size() - 2) * sizeof(float);
        conv1dTiledKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_output, d_signal, signal.size(), mask.size());
    }


    cudaMemcpy(convSignal.data(), d_output, signal.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cudaFree(d_mask);
    cudaFree(d_output);

    return convSignal;
}


std::vector<float> convolutionOnHost(const std::vector<float> &signal, const std::vector<float> &mask)
{
    int signalWidth = static_cast<int>(signal.size());
    int maskWidth = static_cast<int>(mask.size());
    int outputWidth = signalWidth;

    std::vector<float> output(outputWidth, 0.0f);

    // Convolution with zero padding
    int n = maskWidth / 2;
    for (int idxP = 0; idxP < outputWidth; ++idxP)
    {
        float convAccum = 0.0f;
        for (int i = idxP - n; i <= idxP + n; ++i)
        {
            if (i >= 0 && i < signalWidth)
            {
                convAccum += signal[i] * mask[i - (idxP - n)];
            }
        }
        output[idxP] = convAccum;
    }

    return output;
}
