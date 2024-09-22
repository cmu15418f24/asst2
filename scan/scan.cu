#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);

void printCudaInfo();


// TODO unused, delete
__global__ void cuda_full_sequential(int *device_data, int length, int N) {

    printf("\nINSIDE CUDA KERNEL FULL SEQUENTIAL\n");

    // upsweep phase.
    for (int twod = 1; twod < N; twod *= 2) {
        int twod1 = twod * 2;

        // TODO understand how this call differers when threads are in 2D
//            first_loop<<<blockCnt, threadsPerBlock>>>(device_data, N, twod, twod1, numThreads);
//            cudaDeviceSynchronize();
        for(int i = 0; i < N; i += twod1) {
            device_data[i + twod1 - 1] += device_data[i + twod - 1];
        }
    }

    // data[N - 1] = 0;
    int zero = 0;
    device_data[N - 1] = 0;
    device_data[length - 1] = 0;
//    cudaMemcpy(device_data + (length-1), &zero, sizeof(int), cudaMemcpyHostToDevice);

    // downsweep phase.
    for (int twod = N / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;

        for(int i = 0; i < N; i += twod1) {
            int t = device_data[i + twod - 1];
            device_data[i + twod - 1] = device_data[i + twod1 - 1];
            // change twod1 below to twod to reverse prefix sum.
            device_data[i + twod1 - 1] += t;
        }
    }
}



/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void first_loop(int *data, int N, int twod, int twod1, int numThreads) {
    int start = twod1 * (blockIdx.x * blockDim.x + threadIdx.x);
    for(int i = start; i + twod1 - 1 < N; i += twod1  * numThreads) {
//        if ((i + twod1 - 1 < 0) || (i + twod1 - 1 >=N)) {
//            printf("!!!!!!first loop trying to access %d\n", i + twod1 - 1);
//        }
//        if ((i + twod - 1 < 0) || (i + twod - 1 >=N)) {
//            printf("!!!!!!first loop trying to access %d\n", i + twod - 1);
//        }

        if ((i + twod1 - 1 >= 0) && (i + twod1 - 1 < N) && (i + twod - 1 >= 0) && (i + twod - 1
        <N)) {
            data[i + twod1 - 1] += data[i + twod - 1];
        }
    }
}

__global__ void second_loop(int *data, int N, int twod, int twod1, int numThreads) {
    int start = twod1 * (blockIdx.x * blockDim.x + threadIdx.x);

    for(int i = start; i < N; i += twod1 * numThreads) {
//        if ((i + twod1 - 1 < 0) || (i + twod1 - 1 >=N)) {
//            printf("!!!!!!second loop trying to access %d\n", i + twod1 - 1);
//        }
//        if ((i + twod - 1 < 0) || (i + twod - 1 >=N)) {
//            printf("!!!!!!second loop trying to access %d\n", i + twod - 1);
//        }

        if ((i + twod1 - 1 >= 0) && (i + twod1 - 1 < N) && (i + twod - 1 >= 0) && (i + twod - 1
        < N)) {
            int t = data[i + twod - 1];
            data[i + twod - 1] = data[i + twod1 - 1];
            // change twod1 below to twod to reverse prefix sum.
            data[i + twod1 - 1] += t;
        }

    }

}

void exclusive_scan(int *device_data, int length) {

//    printCudaInfo();

    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
//    printf("\nstart exclusive_scan\n");
    int zero = 0;

    int N = nextPow2(length); // TODO round-up or something

    int decision = 0; // 0 for main parallel, 1 for sequential, 2 for threadstest
    if (decision == 0) {
        // upsweep phase.
        for (int twod = 1; twod < N; twod *= 2) {
            int twod1 = twod * 2;

            // TODO understand how this call differers when threads are in 2D
            int threadsPerBlock = 32;
            int blockCnt = 32; // TODO is this what we want?
            int numThreads = threadsPerBlock * blockCnt; // TODO should this be automated?
            first_loop<<<blockCnt, threadsPerBlock>>>(device_data, N, twod, twod1, numThreads);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA ERROR IN FIRST LOOP: %s\n", cudaGetErrorString(err));
                exit(err);
            }
            cudaDeviceSynchronize();
        }

        // data[N - 1] = 0;
        cudaMemcpy(device_data + (N-1), &zero, sizeof(int), cudaMemcpyHostToDevice);
        printf("!! moving on to second loop (downsweep phase) !!!");

        // downsweep phase.
        for (int twod = N / 2; twod >= 1; twod /= 2) {
            int twod1 = twod * 2;

            int threadsPerBlock = 32;
            int blockCnt = 32; // TODO is this what we want?
            int numThreads = threadsPerBlock * blockCnt; // TODO should this be automated?
            second_loop<<<blockCnt, threadsPerBlock>>>(device_data, N, twod, twod1, numThreads);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA ERROR IN SECOND LOOP: %s\n", cudaGetErrorString(err));
                exit(err);
            }
            cudaDeviceSynchronize();
        }

    } else if (decision == 1) {
    } else if (decision == 2) {
    }


//    cudaDeviceSynchronize();
//    printf("end exclusive_scan\n");
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int *inarray, int *end, int *resultarray) {
    int *device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // TODO delete
//    printf("\n100 first Results:\n");
//    for (int i = 0; i < 100; i++) {
//        printf("%d ", resultarray[i]);
//    }
//    printf("\n");

    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}



int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    return 0;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

        printf("deviceProps.maxThreadsPerBlockk: %d\n", deviceProps.maxThreadsPerBlock);
        printf("deviceProp.maxThreadsPerMultiProcessor: %d\n", deviceProps
        .maxThreadsPerMultiProcessor);
    }
    printf("---------------------------------------------------------\n");
}
