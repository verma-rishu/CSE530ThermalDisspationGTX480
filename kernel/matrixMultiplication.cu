#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DEBUG 0

//#define dim

int **matrixA, **matrixB, **matrixC, **matrixD;

__global__ void mult_matrix(int** matrixA, int** matrixB, int** matrixC) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
    int k;
    int dim = sizeof(matrixA);
	if (i < dim && j < dim) {
        for (k = 0; k < dim; k++) {
            matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
        }
    }
}

int main(int argc, char *argv[]) {
    if( argc == 2 ) {
      printf("Block Dimensions input: %s", argv[1]);
     }
   else{
      printf("Invalid input");
    }
    int dim = atoi(argv[1]);
    matrixA = (int**)malloc(sizeof(int*) * dim);
    matrixB = (int**)malloc(sizeof(int*) * dim);
    matrixC = (int**)malloc(sizeof(int*) * dim);
    for(int i = 0; i < dim; i++) {
         matrixA[i]= (int*)malloc(dim);
         matrixB[i]= (int*)malloc(dim);
         matrixC[i]= (int*)malloc(dim);
    }

	int (*deviceA)[dim];
	int (*deviceB)[dim];
	int (*deviceC)[dim];
	int i, j, k;
    
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			matrixA[i][j] = rand() % 100;
			matrixB[i][j] = rand() % 100;
            matrixC[i][j] = 0;
            matrixD[i][j] = 0;
		}
	}
	
	cudaEvent_t start_time, stop_time;
	float elapsedTime;

	cudaMalloc((void **) &deviceA, dim * dim * sizeof(int));
	cudaMalloc((void **) &deviceB, dim * dim * sizeof(int));
	cudaMalloc((void **) &deviceC, dim * dim * sizeof(int));

	cudaMemcpy(deviceA, matrixA, dim * dim * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, matrixB, dim * dim * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32, 32);
	dim3 numOfBlocks(ceil(dim / 32.0), ceil(dim / 32.0));
	cudaEventCreate(&start_time);
	cudaEventRecord(start_time, 0);

	mult_matrix<<<numOfBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceC);
	
	cudaEventCreate(&stop_time);
	cudaEventRecord(stop_time, 0);
	cudaEventSynchronize(stop_time);

	cudaEventElapsedTime(&elapsedTime, start_time, stop_time);
	cudaMemcpy(matrixC, deviceC, dim * dim * sizeof(int), cudaMemcpyDeviceToHost);
    
#if DEBUG
    printf("\nmatrixA-\n");
    for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
                printf("%d\t", matrixA[i][j]);
		}
		printf("\n");
	}
	
    printf("\nmatrixB-\n");
    for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
                printf("%d\t", matrixB[i][j]);
		}
		printf("\n");
	}
    
    printf("\nmatrixC-\n");
    for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
                printf("%d\t", matrixC[i][j]);
		}
		printf("\n");
	}
	printf("\n");
#endif
	
	printf("Parallely Elapsed Time: %f ms\n", elapsedTime);
	
	clock_t start_time_nonparallely, stop_time_nonparallely;
	start_time_nonparallely = clock();
    
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
                matrixD[i][j] += matrixA[i][k] * matrixB[k][j];
            }
		}
	}
	
	stop_time_nonparallely = clock();
	printf("Non-parallely Elapsed Time: %f ms\n", (float) ((stop_time_nonparallely) - (start_time_nonparallely)));
	
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
    
    return 0;
}