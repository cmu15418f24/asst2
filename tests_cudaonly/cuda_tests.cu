//
// Created by hugol on 9/21/2024.
//
#include <stdio.h>

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <driver_functions.h>
//#include "CycleTimer.h"


__global__ void childKernel() {
    printf("Hello ");
}

__global__ void tailKernel() {
    printf("World!\n");
}

__global__ void parentKernel() {
    // launch child
    childKernel<<<1,1>>>();

    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    tailKernel<<<1,1>>>();

    // launch tail into cudaStreamTailLaunch stream
    // implicitly synchronizes: waits for child to complete
//    tailKernel<<<1,1,0,cudaStreamTailLaunch>>>(); # TODO make work
}

int main(int argc, char *argv[]) {

        printf("ENTERING tests/cuda_tests.cu MAIN\n");


        // launch parent
        parentKernel<<<1,1>>>();

        if (cudaSuccess != cudaGetLastError()) {
            return 1;
        }

        // wait for parent to complete
        if (cudaSuccess != cudaDeviceSynchronize()) {
            return 2;
        }

        return 0;
}