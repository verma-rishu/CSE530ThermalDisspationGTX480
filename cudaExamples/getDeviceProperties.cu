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

int main(void){
    cudaDeviceProp property;

    int count;
    cudaGetDeviceCount(&count);
    printf("Count: "+count);
    for(int i=0;i<count;i++){
        cudaGetDeviceProperties(&property, i);
    }
    return 0;
}
