#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>

__global__ void add(int a, int b, int *c){
    *c=a+b;
}

int main(void){
    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    add<<<1,1>>>(2,7,dev_c);
    cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);
    printf("2 + 7=%d\n",c);
    cudaFree(dev_c);
    printf("Hello World!\n");
    return 0;
}
/* 
1. cudaMalloc: tells cuda kernel to allocate memory on the device 
2. host pointers can access memeroty from the host code and device pointers can access memory from device code.
Although we can pass host pointers to the memory but cannot use it to access the memory of the device code.
*/