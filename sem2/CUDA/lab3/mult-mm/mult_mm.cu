#include "mult_mm.h"
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMulKernel(const float *A, const float *B, float *C,
                                int A_rows, int A_cols, int B_cols)
{
    int row =  blockIdx.x*blockDim.x + threadIdx.x;
    int column = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < A_rows && column)
    {
        float value = 0.0f;
        for (int k = 0; k<B_cols; k++)
        {
            value += A[row*A_cols + k]*B[k*B_cols + column];
        }
        C[row*A_cols + column] = value;
    }

}

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C,
                                     int A_rows, int A_cols, int B_cols)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int column = blockIdx.x*blockDim.x + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (A_cols + TILE_SIZE - 1)/TILE_SIZE; ++t)
    {

        if (row < A_rows && t*TILE_SIZE*threadIdx.x < A_cols)
            tileA[threadIdx.x][threadIdx.y] = A[row*A_cols + t*TILE_SIZE+threadIdx.x];
        else
            tileA[threadIdx.x][threadIdx.y] = 0.0f;

        if (column < B_cols && t*TILE_SIZE+threadIdx.y < A_cols)
            tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*B_cols + column];
        else
            tileA[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++)
            value += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < A_rows && column < B_cols)
        C[row*A_cols + column] = value;
}

__global__ void matrixMulGranularKernel(const float *A, const float *B, float *C,
                                        int A_rows, int A_cols, int B_cols)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB1[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB2[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col1 = blockIdx.x * blockDim.x + threadIdx.x;
    int col2 = col1 + blockDim.x;

    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (int t = 0; t < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        if (row < A_rows && (t * TILE_SIZE + threadIdx.x) < A_cols)
            tileA[threadIdx.y][threadIdx.x] = A[row * A_cols + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < A_cols) {
            if (col1 < B_cols)
                tileB1[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * B_cols + col1];
            else
                tileB1[threadIdx.y][threadIdx.x] = 0.0f;

            if (col2 < B_cols)
                tileB2[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * B_cols + col2];
            else
                tileB2[threadIdx.y][threadIdx.x] = 0.0f;
        } else {
            tileB1[threadIdx.y][threadIdx.x] = 0.0f;
            tileB2[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();


        for (int k = 0; k < TILE_SIZE; ++k)
        {
            acc1 += tileA[threadIdx.y][k] * tileB1[k][threadIdx.x];
            acc2 += tileA[threadIdx.y][k] * tileB2[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < A_rows) {
        if (col1 < B_cols)
            C[row * B_cols + col1] = acc1;
        if (col2 < B_cols)
            C[row * B_cols + col2] = acc2;
    }
}

Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method)
{
    Matrix outputMatrix(A.getRows(), B.getCols());

    float* d_AMatrix;
    float* d_BMatrix;
    float* d_CMatrix;

    size_t sizeA = A.getRows() * A.getCols() * sizeof(float);
    size_t sizeB = B.getRows() * B.getCols() * sizeof(float);
    size_t sizeC = A.getRows() * B.getCols() * sizeof(float);

    cudaMalloc((void**)&d_AMatrix, sizeA);
    cudaMalloc((void**)&d_BMatrix, sizeB);
    cudaMalloc((void**)&d_CMatrix, sizeC);

    cudaMemcpy(d_AMatrix, A.getDataConstPtr(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BMatrix, B.getDataConstPtr(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B.getCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (A.getRows() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    switch (method)
    {
        case MultMethod::Standard:
            std::cout << "debug1";
            matrixMulKernel<<<numBlocks, threadsPerBlock>>>(
                d_AMatrix, d_BMatrix, d_CMatrix,
                A.getRows(), A.getCols(), B.getCols());
            break;

        case MultMethod::Tiled:
            std::cout << "debug2";
            matrixMulTiledKernel<<<numBlocks, threadsPerBlock>>>(
                d_AMatrix, d_BMatrix, d_CMatrix,
                A.getRows(), A.getCols(), B.getCols());
            break;

        default:
            std::cout << "debug2";
            std::cout << "Nieznana metoda mnożenia macierzy!" << std::endl;
            break;
    }

    cudaMemcpy(outputMatrix.getDataPtr(), d_CMatrix, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_AMatrix);
    cudaFree(d_BMatrix);
    cudaFree(d_CMatrix);

    return outputMatrix;
}


Matrix multMatrixMatrixOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < B.getCols(); ++j)
        {
            for (unsigned int k = 0; k < A.getCols(); ++k)
            {
                C.getDataPtr()[i * C.getCols() + j] +=
                    A.getDataConstPtr()[i * A.getCols() + k] *
                    B.getDataConstPtr()[k * B.getCols() + j];
            }
        }
    }
    return C;
}
