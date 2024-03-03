/*
 * Copyright 2021-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample to show how to correlate CUDA APIs with the corresponding GPU
 * activities using the correlation-id field in the activity records.
 *
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
using namespace std;

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Macros
#define COMPUTE_N 50000

// Map to store correlation id and its corresponding activity record.
static std::map<uint32_t, CUpti_Activity *> s_CorrelationMap;
static std::map<uint32_t, CUpti_Activity *> s_ConnectionMap;

// Iterator to traverse the map.
static std::map<uint32_t, CUpti_Activity *>::iterator s_Iter;

// Kernels
__global__ void
VectorAdd(
    const int* pA,
    const int* pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

__global__ void
VectorSub(
    const int* pA,
    const int* pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] - pB[i];
    }
}

static void
DoPass(
    cudaStream_t stream)
{
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    size_t size = COMPUTE_N * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;

    // Allocate input vectors pHostA and pHostB in host memory.
    // Don't bother to initialize.
    pHostA = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int*)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice, stream));

    blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (pDeviceA, pDeviceB, pDeviceC, COMPUTE_N);
    RUNTIME_API_CALL(cudaGetLastError());

    VectorSub <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (pDeviceA, pDeviceB, pDeviceC, COMPUTE_N);
    RUNTIME_API_CALL(cudaGetLastError());

    RUNTIME_API_CALL(cudaMemcpyAsync(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost, stream));

    if (stream == 0)
    {
        RUNTIME_API_CALL(cudaDeviceSynchronize());
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    }

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

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }
    if (pDeviceC)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }
}

static void
PrintCorrealtionInformation()
{
    s_Iter = s_CorrelationMap.begin();
    // Iterate over the map using Iterator till end.
    while (s_Iter != s_CorrelationMap.end())
    {
        // s_Iter->first is correlation id and s_Iter->second is activity record.
        switch (s_Iter->second->kind) {
            case CUPTI_ACTIVITY_KIND_MEMCPY:
            {
                CUpti_ActivityMemcpy5 *pMemcpyRecord = (CUpti_ActivityMemcpy5 *)s_Iter->second;
                // Check whether memcpy correlation id is present in connection map.
                if (s_ConnectionMap.find(pMemcpyRecord->correlationId) != s_ConnectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", pMemcpyRecord->correlationId);

                    PrintActivity(s_Iter->second, stdout);

                    CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)s_ConnectionMap[pMemcpyRecord->correlationId];
                    if (pApiRecord != NULL)
                    {
                        PrintActivity((CUpti_Activity *)pApiRecord, stdout);
                    }
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET:
            {
                CUpti_ActivityMemset4 *pMemsetRecord = (CUpti_ActivityMemset4 *)s_Iter->second;
                // Check whether memset correlation id is present in connection map.
                if (s_ConnectionMap.find(pMemsetRecord->correlationId) != s_ConnectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", pMemsetRecord->correlationId);

                    PrintActivity(s_Iter->second, stdout);

                    CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)s_ConnectionMap[pMemsetRecord->correlationId];
                    if (pApiRecord != NULL)
                    {
                        PrintActivity((CUpti_Activity *)pApiRecord, stdout);
                    }
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            {
                CUpti_ActivityKernel9 *pKernelRecord = (CUpti_ActivityKernel9 *)s_Iter->second;
                // Check whether kernel correlation id is present in connection map.
                if (s_ConnectionMap.find(pKernelRecord->correlationId) != s_ConnectionMap.end())
                {
                    printf("\nCUDA_API AND GPU ACTIVITY CORRELATION : correlation %u\n", pKernelRecord->correlationId);

                    PrintActivity(s_Iter->second, stdout);

                    CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)s_ConnectionMap[pKernelRecord->correlationId];
                    if (pApiRecord != NULL)
                    {
                        PrintActivity((CUpti_Activity *)pApiRecord, stdout);
                    }

                }
                break;
            }
            default:
                break;
        }

        s_Iter++;
    }
}

// Store the runtime and driver API records in s_ConnectionMap and others in s_CorrelationMap.
void
CorrelationActivityRecords(
    CUpti_Activity *pRecord)
{
    switch (pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy5 *pMemcpyRecord = (CUpti_ActivityMemcpy5 *)pRecord;
            uint32_t recordSize = sizeof(CUpti_ActivityMemcpy5);

            // Creating copy of the record to print correaltion data after the application is over.
            CUpti_Activity *pRecordCopy = (CUpti_Activity *)malloc(recordSize);
            MEMORY_ALLOCATION_CALL(pRecordCopy);
            memset(pRecordCopy, 0, recordSize);
            memcpy(pRecordCopy, pRecord, recordSize);

            s_CorrelationMap[pMemcpyRecord->correlationId] = pRecordCopy;

            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET:
        {
            CUpti_ActivityMemset4 *pMemsetRecord = (CUpti_ActivityMemset4 *)pRecord;
            uint32_t recordSize = sizeof(CUpti_ActivityMemset4);

            // Creating copy of the record to print correaltion data after the application is over.
            CUpti_Activity *pRecordCopy = (CUpti_Activity *)malloc(recordSize);
            MEMORY_ALLOCATION_CALL(pRecordCopy);
            memset(pRecordCopy, 0, recordSize);
            memcpy(pRecordCopy, pRecord, recordSize);

            s_CorrelationMap[pMemsetRecord->correlationId] = pRecordCopy;
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel9 *pKernelRecord = (CUpti_ActivityKernel9 *)pRecord;
            uint32_t recordSize = sizeof(CUpti_ActivityKernel9);

            // Creating copy of the record to print correaltion data after the application is over.
            CUpti_Activity *pRecordCopy = (CUpti_Activity *)malloc(recordSize);
            MEMORY_ALLOCATION_CALL(pRecordCopy);
            memset(pRecordCopy, 0, recordSize);
            memcpy(pRecordCopy, pRecord, recordSize);

            s_CorrelationMap[pKernelRecord->correlationId] = pRecordCopy;
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_DRIVER:
        {
            CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)pRecord;
            uint32_t recordSize = sizeof(CUpti_ActivityAPI);

            // Creating copy of the record to print correaltion data after the application is over.
            CUpti_Activity *pRecordCopy = (CUpti_Activity *)malloc(recordSize);
            MEMORY_ALLOCATION_CALL(pRecordCopy);
            memset(pRecordCopy, 0, recordSize);
            memcpy(pRecordCopy, pRecord, recordSize);

            s_ConnectionMap[pApiRecord->correlationId] = pRecordCopy;
            break;
        }
        default:
            break;
    }
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = CorrelationActivityRecords;
    pUserData->printActivityRecords        = 0;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable CUDA runtime activity kinds for CUPTI to provide correlation ids.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));

    // Enable activity kinds to trace GPU activities kernel and memcpy.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
}

int
main(
    int argc,
    char *argv[])
{
    CUdevice device;
    char deviceName[256];

    SetupCupti();

    // Initialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
    printf("Device Name: %s\n", deviceName);
    RUNTIME_API_CALL(cudaSetDevice(0));

    // DoPass with default stream.
    DoPass(0);

    // DoPass with user stream.
    cudaStream_t stream;
    RUNTIME_API_CALL(cudaStreamCreate(&stream));
    DoPass(stream);
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    DeInitCuptiTrace();

    PrintCorrealtionInformation();

    exit(EXIT_SUCCESS);
}
