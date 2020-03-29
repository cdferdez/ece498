#include <wb.h>
#include <math.h>
#define TILE_WIDTH 32
#define BLOCK_WIDTH 32
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  // retrieve row and col index
  int Row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  int Col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
  
  // check bounds of matrices
  if ((Row < numCRows) && (Col < numCColumns)) {
    float Pvalue = 0;

    for (int k = 0; k < numAColumns; ++k) {
      Pvalue += A[Row*numAColumns + k] * B[k*numBColumns + Col];
    }

    C[Row*numCColumns + Col] = Pvalue;
  }
}

// tiled matrix multiplication implementation
__global__ void tiledMatrixMultiply(float *A, float *B, float *C, int numARows,
                                    int numAColumns, int numBRows,
                                    int numBColumns, int numCRows,
                                    int numCColumns) {
  // tiled matrix multiplication
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  // get thread index info
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  
  // obtain global row and global column
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  float Pvalue = 0;

  // check bounds
  for (int m = 0; m < (numAColumns-1)/TILE_WIDTH + 1; ++m) {
    // more bound checking 
    if (Row < numARows && m*TILE_WIDTH+tx < numAColumns) {
      subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH + tx];
    } else {
      subTileA[ty][tx] = 0.0;
    }

    if (m*TILE_WIDTH+ty < numBRows && Col < numBColumns) {
      subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns + Col];  
    } else {
      subTileB[ty][tx] = 0.0;
    }
    
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    }
  }

  __syncthreads();

  if (Row < numCRows && Col < numCColumns) {
    C[Row*numCColumns + Col] = Pvalue;     
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int blockWidth = 32;
  float blockXDim = (float)numCColumns / (float)blockWidth;
  float blockYDim = (float)numCRows / (float)blockWidth;
  
  //wbLog(TRACE, ceil(blockYDim), " ", ceil(blockXDim));
  
  dim3 dimGrid(ceil(blockXDim), ceil(blockYDim), 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, 
                                      numARows, numAColumns, 
                                      numBRows, numBColumns,
                                      numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
  
  //wbLog(TRACE, (numAColumns - 1)*TILE_WIDTH + 1);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
