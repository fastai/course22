/*
 * Copyright 2014-2022 NVIDIA Corporation. All rights reserved.
 *
 * Sample CUPTI app to demonstrate the usage of unified memory counter profiling
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

template<class T>
__host__ __device__ void
CheckData(
    const char *pLoc,
    T *pData,
    int size,
    int expectedValue)
{
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++)
    {
        if (pData[i] != expectedValue)
        {
            printf("Mismatch found on %s\n", pLoc);
            printf("Address 0x%p, Observed = 0x%x Expected = 0x%x\n", (pData + i), pData[i], expectedValue);
            break;
        }
    }
}

template<class T>
__host__ __device__ void
WriteData(
    T *pData,
    int size,
    int writeValue)
{
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++)
    {
        pData[i] = writeValue;
    }
}

__global__ void
TestKernel(
    int *pData,
    int size,
    int expectedValue)
{
    CheckData("GPU", pData, size, expectedValue);
    WriteData(pData, size, -expectedValue);
}

// Functions
static void
SetupCupti()
{
    CUptiResult cuptiResult;
    CUpti_ActivityUnifiedMemoryCounterConfig config[2];

    // Configure Unified memory counters.
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    cuptiResult = cuptiActivityConfigureUnifiedMemoryCounter(config, 2);
    if (cuptiResult == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED)
    {
        printf("Warning: Unified memory is not supported on the underlying platform. Waiving sample.\n");
        exit(EXIT_WAIVED);
    }
    else if (cuptiResult == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE)
    {
        printf("Warning: Unified memory is not supported on the device. Waiving sample.\n");
        exit(EXIT_WAIVED);
    }
    else if (cuptiResult == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES)
    {
        printf("Warning: Unified memory is not supported on the non-P2P multi-gpu setup. Waiving sample.\n");
        exit(EXIT_WAIVED);
    }
    else
    {
        CUPTI_API_CALL(cuptiResult);
    }

    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable unified memory counter activity.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
}

int
main(
    int argc,
    char *argv[])
{
    int deviceCount;
    int *pData = NULL;
    int size = 64 * 1024;     // 64 KB
    int i = 123;

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA. Waiving sample.\n");
        exit(EXIT_WAIVED);
    }

    SetupCupti();

    // Allocate unified memory.
    printf("Allocation size in bytes %d\n", size);
    RUNTIME_API_CALL(cudaMallocManaged(&pData, size));

    // CPU access.
    WriteData(pData, size, i);

    // Kernel launch.
    TestKernel<<< 1, 1 >>> (pData, size, i);
    RUNTIME_API_CALL(cudaGetLastError());
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // CPU access.
    CheckData("CPU", pData, size, -i);

    // Free unified memory.
    RUNTIME_API_CALL(cudaFree(pData));

    // Disable unified memory counter activity.
    CUPTI_API_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

    RUNTIME_API_CALL(cudaDeviceReset());

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
