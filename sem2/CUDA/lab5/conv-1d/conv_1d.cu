#include "conv_1d.h"

__constant__ float c_mask[MAX_MASK_WIDTH];

__global__ void conv1dBasicKernel(float *output, const float *signal, const float *mask, const int width, const int maskWidth)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;

    if (x < width)
    {
        float sum = 0;
        int halfMask = maskWidth/2;

        for (int i = 0; i <= maskWidth; i++)
        {
            int signalIdx = x + i - halfMask;
            if (signalIdx >= 0 && signalIdx <= width){
                sum += signal[signalIdx]*mask[i];
            }
        }

        output[x] = sum;
    }
}

__global__ void conv1dTiledKernel(float *output, const float *signal, const int width, const int maskWidth)
{
    extern __shared__ float sharedSignal[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    int start = bx*bdx;
    int x = start + tx;

    int radius = maskWidth/2;

    int sharedIdx = tx+radius;

    if (x < width)
        sharedSignal[sharedIdx] = signal[x];
    else
        sharedSignal[sharedIdx] = 0.0f;
    
    if (tx < radius){
        int leftIdx = x - radius;
        sharedSignal[leftIdx] = (leftIdx >= 0) ? signal[leftIdx] : 0.0f;
    };

    if (tx >= bdx - radius){
        int rightIdx = x + radius;
        if (rightIdx < width)
            sharedSignal[sharedIdx + radius] = signal[rightIdx];
        else
            sharedSignal[sharedIdx + radius] = 0.0f;
    }

    __syncthreads();

    if (x < width){
        float sum = 0.0f;

        for (int k = 0; k<maskWidth; ++k){
            sum += sharedSignal[tx+k] * c_mask[k];
        }
        output[x] = sum;
    }

}

std::vector<float> convolutionOnDevice(const std::vector<float> &signal, const std::vector<float> &mask, ConvMethod method)
{
    float* d_signal;
    float* d_mask;
    float* out_signal;

    std::vector<float> convSignal;

    cudaMalloc((void**)&d_signal, signal.size()*sizeof(float));
    cudaMalloc((void**)&out_signal, signal.size()*sizeof(float));
    cudaMalloc((void**)&d_mask, mask.size()*sizeof(float));

    cudaMemcpy(d_signal, signal.data(), signal.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), mask.size()*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (signal.size() + threadsPerBlock - 1) / threadsPerBlock;

    if (method == ConvMethod::Basic)
        conv1dBasicKernel<<<threadsPerBlock, blocks>>>(out_signal, d_signal, d_mask, signal.size(), mask.size());

    cudaMemcpy(convSignal.data(), out_signal, signal.size()*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cudaFree(d_mask);
    cudaFree(out_signal);

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
