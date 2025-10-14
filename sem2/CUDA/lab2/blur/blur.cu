#include "blur.h"

#define TILE_WIDTH 16

__global__ void blurKernel(float *out, float *in, int width, int height, int blurSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int half = blurSize / 2;

    float sum = 0.0f;
    int count = 0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += in[ny * width + nx];
                count++;
            }
        }
    }

    // średnia z sąsiadów
    out[y * width + x] = sum / count;
}


Image imageBlurOnDevice(const Image &inputImage, int blurSize)
{
    Image outputImage(inputImage.getWidth(), inputImage.getHeight(), true);

    // allocate input and output images in the device
    float *d_inputImage;
    float *d_outputImage;
    cudaMalloc((void **)&d_inputImage, inputImage.getRows() * inputImage.getCols() * sizeof(float));
    cudaMalloc((void **)&d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float));

    cudaMemcpy(d_inputImage, inputImage.getDataConstPtr(), inputImage.getRows() * inputImage.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)outputImage.getCols() / blurSize), ceil((float)outputImage.getRows() / blurSize));
    dim3 dimBlock(blurSize, blurSize, 1);

    blurKernel<<<dimGrid, dimBlock>>>(d_outputImage, d_inputImage, 3, outputImage.getCols(), outputImage.getRows());

    cudaMemcpy(outputImage.getDataPtr(), d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return outputImage;
}

Image imageBlurOnHost(const Image &inputImage, int blurSize)
{
    Image outputImage(inputImage.getWidth(), inputImage.getHeight(), inputImage.isGray());

    int contextRadius = (blurSize - 1) / 2;

    for (unsigned int y = 0; y < inputImage.getHeight(); ++y)
    {
        for (unsigned int x = 0; x < inputImage.getWidth(); ++x)
        {
            float outVal = 0.0f;
            // Inside full context space
            if (x >= contextRadius && x < inputImage.getWidth() - contextRadius && y >= contextRadius && y < inputImage.getHeight() - contextRadius)
            {
                float accumVal = 0.0f;
                for (int c = -contextRadius; c <= contextRadius; c++)
                {
                    for (int r = -contextRadius; r <= contextRadius; ++r)
                    {
                        int accumIdx = (y + c) * inputImage.getWidth() + (x + r);
                        accumVal += inputImage.getDataConstPtr()[accumIdx];
                    }
                }
                outVal = accumVal / (blurSize * blurSize);
            }

            int outIdx = y * inputImage.getWidth() + x;
            outputImage.getDataPtr()[outIdx] = outVal;
        }
    }

    return outputImage;
}
