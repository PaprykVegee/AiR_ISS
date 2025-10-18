#include "mult_mv.h"

__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int nrows, unsigned int ncols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows)
    {
        float sum = 0.0f;

        for (int j = 0; j < ncols; j++)
        {
            sum += A[row*ncols + j] * x[j];
        }

        b[row] = sum;
    }
}

Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x)
{
    Matrix outputMatrix(A.getRows(), 1);  // wynik Mx1

    float* d_inputMatrix;
    float* d_inputVector;
    float* d_outputVector;

    cudaMalloc((void**)&d_inputMatrix, A.getCols() * A.getRows() * sizeof(float));
    cudaMalloc((void**)&d_inputVector, x.getCols() * x.getRows() * sizeof(float));
    cudaMalloc((void**)&d_outputVector, A.getRows() * sizeof(float));

    cudaMemcpy(d_inputMatrix, A.getDataConstPtr(),
               A.getCols() * A.getRows() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_inputVector, x.getDataConstPtr(),
               x.getCols() * x.getRows() * sizeof(float),
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (A.getRows() + threadsPerBlock - 1) / threadsPerBlock;

    multMatrixVector<<<blocksPerGrid, threadsPerBlock>>>(
        d_outputVector, d_inputMatrix, d_inputVector,
        A.getRows(), A.getCols()
    );

    cudaMemcpy(outputMatrix.getDataPtr(), d_outputVector,
               A.getRows() * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inputMatrix);
    cudaFree(d_inputVector);
    cudaFree(d_outputVector);

    return outputMatrix;
}


Matrix multMatrixVectorOnHost(const Matrix &A, const Matrix &x)
{
    if (A.getCols() != x.getRows())
    {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    Matrix b(A.getRows(), 1);
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        float sum = 0.0f;
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            sum += A.getDataConstPtr()[i * A.getCols() + j] * x.getDataConstPtr()[j];
        }
        b.getDataPtr()[i] = sum;
    }
    return b;
}
