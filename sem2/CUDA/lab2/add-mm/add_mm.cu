#include "add_mm.h"

__global__ void addMatricesByElements(const float *A, const float *B, float *C, int ncols, int nrows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x < ncols && y < nrows)
    {
        int idx = y * ncols + x;   
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void addMatricesByRows(const float *A, const float *B, float *C, int ncols, int nrows)
{                 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nrows)
    {
        for (int col = 0; col < ncols; col++)
        {
            int idx = row*ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void addMatricesByColumns(const float *A, const float *B, float *C, int ncols, int nrows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ncols)
    {
        for (int row = 0; row < nrows; row++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}


Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method)
{
    Matrix outputMatrix(A.getCols(), A.getRows());
    size_t size = A.getCols() * A.getRows() * sizeof(float);

    float *d_A_Matrix, *d_B_Matrix, *d_C_Matrix;
    cudaMalloc(&d_A_Matrix, size);
    cudaMalloc(&d_B_Matrix, size);
    cudaMalloc(&d_C_Matrix, size);

    cudaMemcpy(d_A_Matrix, A.getDataConstPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_Matrix, B.getDataConstPtr(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((A.getCols() + blockDim.x - 1) / blockDim.x,
                 (A.getRows() + blockDim.y - 1) / blockDim.y);

    addMatricesByElements<<<gridDim, blockDim>>>(d_A_Matrix, d_B_Matrix, d_C_Matrix,
                                                 A.getCols(), A.getRows());

    cudaMemcpy(outputMatrix.getDataPtr(), d_C_Matrix, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A_Matrix);
    cudaFree(d_B_Matrix);
    cudaFree(d_C_Matrix);

    return outputMatrix;
}


Matrix addMatricesOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    Matrix C(A.getRows(), A.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            C.getDataPtr()[i * A.getCols() + j] = A.getDataConstPtr()[i * A.getCols() + j] + B.getDataConstPtr()[i * A.getCols() + j];
        }
    }
    return C;
}
