/*
 * Copyright 2021-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to show how to do external correlation.
 * The sample pushes external ids in a simple vector addition
 * application showing how you can externally correlate different
 * phases of the code. In this sample it is broken into
 * initialization, execution and cleanup showing how you can
 * correlate all the APIs invloved in these 3 phases in the app.
 *
 * Psuedo code:
 * cuptiActivityPushExternalCorrelationId()
 * ExternalAPI() -> (runs bunch of CUDA APIs/ launches activity on GPU)
 * cuptiActivityPopExternalCorrelationId()
 * All CUDA activity activities within this range will generate external correlation
 * record which then can be used to correlate it with the external API
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Enum mapping the external id to the different phases in the vector addition it correlates to.
typedef enum ExternalId_st
{
    INITIALIZATION_EXTERNAL_ID = 0,
    EXECUTION_EXTERNAL_ID = 1,
    CLEANUP_EXTERNAL_ID = 2,
    MAX_EXTERNAL_ID = 3
} ExternalId;

// Map to book-keep the correlation ids mapped to the external ids.
static std::map<uint64_t, std::vector<uint32_t> > s_externalCorrelationMap;

// Kernels
__global__ void
VectorAdd(const int* pA, const int* pB, int* pC, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        pC[idx] = pA[idx] + pB[idx];
    }
}

// Functions
static void
InitializeVector(int *pVector, int N)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

static void
DoVectorAddition() {
    CUcontext context = 0;
    CUdevice device = 0;

    int N = 50000;
    size_t size = N * sizeof (int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;

    int *pHostA = 0, *pHostB = 0, *pHostC = 0;
    int *pDeviceA = 0, *pDeviceB = 0, *pDeviceC = 0;
    uint64_t id = 0;

    DRIVER_API_CALL(cuDeviceGet(&device, 0));

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    // Push external id for the initialization: memory allocation and memcpy operations from host to device.
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(INITIALIZATION_EXTERNAL_ID)));

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Pop the external id.
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

    // Push external id for the vector addition and copy of results from device to host.
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(EXECUTION_EXTERNAL_ID)));

    // Invoke kernel.
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());
    DRIVER_API_CALL(cuCtxSynchronize());

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Pop the external id.
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

    // Push external id for the cleanup phase in the code.
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(CLEANUP_EXTERNAL_ID)));

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

    // Pop the external id.
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

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

    DRIVER_API_CALL(cuCtxSynchronize());
    DRIVER_API_CALL(cuCtxDestroy(context));
}

static void
ShowExternalCorrelation()
{
    // Print the summary of extenal ids mapping with the correaltion ids.
    printf ("\n=== SUMMARY ===");
    for (auto externalIdIter : s_externalCorrelationMap)
    {
        printf("\nExternal Id: %llu: ", (unsigned long long)externalIdIter.first);

        if (externalIdIter.first == INITIALIZATION_EXTERNAL_ID)
        {
            printf("INITIALIZATION_EXTERNAL_ID");
        }
        else if (externalIdIter.first == EXECUTION_EXTERNAL_ID)
        {
            printf("EXECUTION_EXTERNAL_ID");
        }
        else if (externalIdIter.first == CLEANUP_EXTERNAL_ID)
        {
            printf("CLEANUP_EXTERNAL_ID");
        }

        printf("\n  Correlate to CUPTI records with correlation ids: ");

        auto correlationIter = externalIdIter.second;
        for (auto correlationId : correlationIter)
        {
            printf("%u ", correlationId);
        }

        printf("\n");
    }
}

void
ExternalCorrelationRecords(
    CUpti_Activity *pRecord)
{
    switch(pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
        {
            CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;

            //  Map the correlation ids to external ids for correlation.
            auto externalMapIter = s_externalCorrelationMap.find(pExternalCorrelationRecord->externalId);
            if (externalMapIter == s_externalCorrelationMap.end())
            {
                std::vector<uint32_t> correlationVector;
                correlationVector.push_back((uint32_t)pExternalCorrelationRecord->correlationId);
                s_externalCorrelationMap.insert({(uint64_t)pExternalCorrelationRecord->externalId, correlationVector });
            }
            else
            {
                s_externalCorrelationMap[(uint64_t)pExternalCorrelationRecord->externalId].push_back((uint32_t)pExternalCorrelationRecord->correlationId);
            }

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
    pUserData->pPostProcessActivityRecords = ExternalCorrelationRecords;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable CUDA runtime activity kinds for CUPTI to provide correlation ids.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));

    // Enable external correlation activtiy kind..
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

    // Enable activity kinds to trace GPU activities kernel and memcpy.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
}

int
main(
    int argc,
    char *argv[])
{
    SetupCupti();

    // Initialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    // External correlation with vector addition.
    DoVectorAddition();

    DeInitCuptiTrace();

    ShowExternalCorrelation();

    exit(EXIT_SUCCESS);
}