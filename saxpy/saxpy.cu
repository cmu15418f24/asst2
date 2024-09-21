#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

// TODO
//Question: Compare and explain the difference between the results provided by two sets of timers (the timer
//        you added and the timer that was already in the provided starter code). Are the bandwidth values observed
//        roughly consistent with the reported bandwidths available to the different components of the machine?
//Hint: You should use the web to track down the memory bandwidth of an NVIDIA RTX 2080 GPU, and
//the maximum transfer speed of the computer’s PCIe-x16 bus. It’s PCIe 3.0, and a 16 lane bus connecting
//        the CPU with the GPU.

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    //
    // TODO allocate device memory buffers on the GPU using cudaMalloc
    //
    int byte_count = N * sizeof(float);
    cudaMalloc((void **)&device_x, byte_count);
    cudaMalloc((void **)&device_y, byte_count);
    cudaMalloc((void **)&device_result, byte_count);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_x, xarray, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, byte_count, cudaMemcpyHostToDevice);

    // run kernel
    double start_time_saxpy_kernel = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();
    double end_time_saxpy_kernel = CycleTimer::currentSeconds();
    double kernelDuration = end_time_saxpy_kernel - start_time_saxpy_kernel;
    printf("\nKernel Only: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes,
                                                                               kernelDuration));
    //
    // TODO copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(resultarray, device_result, byte_count, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "\nWARNING: A CUDA error occured: code=%d, %s\n", errCode,
                cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("\nOverall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes,
                                                                                overallDuration));

    // TODO free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
