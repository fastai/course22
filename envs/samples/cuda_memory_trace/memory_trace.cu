/*
 * Copyright 2021-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print trace of CUDA memory operations.
 * The sample also traces CUDA memory operations done via
 * default memory pool.
 *
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Kernels
__global__ void
VectorAdd(
    const float *pA,
    const float *pB,
    float *pC,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

// Functions
static void
DoMemoryAllocations()
{
    cudaDeviceProp deviceProperties;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&deviceProperties, 0));

    if (!deviceProperties.managedMemory)
    {
        // This samples requires being run on a device that supports Unified Memory.
        printf("Warning: Unified Memory not supported on this device. Waiving sample.\n");
        return;
    }

    int nElements = 1048576;
    size_t size = nElements * sizeof(int);

    int *pHostA, *pHostB;
    int *pDeviceA, *pDeviceB;

    // Allocate memory.
    RUNTIME_API_CALL(cudaMallocHost((void **)&pHostA, size));
    RUNTIME_API_CALL(cudaHostAlloc((void **)&pHostB, size, cudaHostAllocPortable));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMallocManaged((void **)&pDeviceB, size, cudaMemAttachGlobal));

    // Free the allocated memory.
    RUNTIME_API_CALL(cudaFreeHost(pHostA));
    RUNTIME_API_CALL(cudaFreeHost(pHostB));
    RUNTIME_API_CALL(cudaFree(pDeviceA));
    RUNTIME_API_CALL(cudaFree(pDeviceB));
}

static void
DoMemoryAllocationsViaMemoryPool()
{
    int nElements = 1048576;
    size_t bytes = nElements * sizeof(float);

    float *pHostA, *pHostB, *pHostC;
    float *pDeviceA, *pDeviceB, *pDeviceC;
    cudaStream_t stream;

    int isMemPoolSupported = 0;
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, 0);
    // For enhance compatibility cases, the attribute cudaDevAttrMemoryPoolsSupported might not be present.
    // return early if Runtime API does not return cudaSuccess.
    if (!isMemPoolSupported || cudaStatus != cudaSuccess)
    {
        printf("Warning: Memory pool not supported on this device. Waiving sample.\n");
        return;
    }

    // Allocate and initialize memory on host and device.
    pHostA = (float*) malloc(bytes);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (float*) malloc(bytes);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (float*) malloc(bytes);
    MEMORY_ALLOCATION_CALL(pHostC);

    for (int n = 0; n < nElements; n++)
    {
        pHostA[n] = rand() / (float)RAND_MAX;
        pHostB[n] = rand() / (float)RAND_MAX;
    }

    RUNTIME_API_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Allocate memory using default memory pool.
    RUNTIME_API_CALL(cudaMallocAsync(&pDeviceA, bytes, stream));
    RUNTIME_API_CALL(cudaMallocAsync(&pDeviceB, bytes, stream));
    RUNTIME_API_CALL(cudaMallocAsync(&pDeviceC, bytes, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, bytes, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nElements/(float)block.x));
    VectorAdd<<<grid, block, 0, stream>>>(pDeviceA, pDeviceB, pDeviceC, nElements);

    // Free the allocated memory.
    RUNTIME_API_CALL(cudaFreeAsync(pDeviceA, stream));
    RUNTIME_API_CALL(cudaFreeAsync(pDeviceB, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pHostC, pDeviceC, bytes, cudaMemcpyDeviceToHost, stream));
    RUNTIME_API_CALL(cudaFree(pDeviceC));

    RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    RUNTIME_API_CALL(cudaStreamDestroy(stream));

    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }
    if (pHostC)
    {
        free(pHostC);
    }
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable CUPTI activities related to memory allocation
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY_POOL));

}

int
main(
    int argc,
    char *argv[])
{
    SetupCupti();

    // Intialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    char deviceName[256];
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
    printf("Device Name: %s\n", deviceName);
    RUNTIME_API_CALL(cudaSetDevice(0));

    DoMemoryAllocations();
    DoMemoryAllocationsViaMemoryPool();

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
