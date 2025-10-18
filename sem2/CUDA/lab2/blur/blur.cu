#include "blur.h"

#define TILE_WIDTH 16

__global__ void blurKernel(float *out, float *in, int width, int height, int blurSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int contextRadius = (blurSize - 1) / 2;
    float accumVal = 0.0f;
    int count = 0;

    if (x >= contextRadius && x < width - contextRadius && y >= contextRadius && y < height - contextRadius)
    {
        for (int dy = -contextRadius; dy <= contextRadius; dy++)
        {
            for (int dx = -contextRadius; dx <= contextRadius; dx++)
            {
                int ix = x + dx;
                int iy = y + dy;
                int index = iy * width + ix;

                accumVal += in[index];
                count++;
            }
        }

        out[y * width + x] = accumVal / count;
    }
    else
    {
        out[y * width + x] = 0.0f;
    }

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

    dim3 dimGrid(ceil((float)outputImage.getCols() / TILE_WIDTH),
                ceil((float)outputImage.getRows() / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    blurKernel<<<dimGrid, dimBlock>>>(d_outputImage, d_inputImage,
        outputImage.getCols(), outputImage.getRows(), blurSize);

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
