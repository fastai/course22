/*
 * Copyright 2010-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain timestamps
 * using callbacks for CUDA runtime APIs
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
#include "cupti.h"
#include "helper_cupti.h"

// Structure to hold data collected by callback.
typedef struct RuntimeApiTrace_st
{
    const char          *pFunctionName;
    uint64_t            startTimestamp;
    uint64_t            endTimestamp;
    size_t              memcpyBytes;
    enum cudaMemcpyKind memcpyKind;
} RuntimeApiTrace;

enum launchOrder
{
    MEMCPY_H2D1,
    MEMCPY_H2D2,
    MEMCPY_D2H,
    KERNEL,
    THREAD_SYNC,
    LAUNCH_LAST
};

// Kernel
// Vector addition kernel.
__global__ void
VectorAdd(
    const int *pA,
    const int *pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

// Functions

// Initialize a vector.
static void
InitializeVector(
    int *pVector,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

void CUPTIAPI
TimestampCallback(
    void *pUserdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const CUpti_CallbackData *pCallbackData)
{
    static int s_MemTransCount = 0;
    uint64_t startTimestamp;
    uint64_t endTimestamp;
    RuntimeApiTrace *pRuntimeApiTrace = (RuntimeApiTrace*)pUserdata;

    // Data is collected only for the following API.
    if ((callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) ||
        (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020) ||
        (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020))
    {

        // Set pointer depending on API.
        if ((callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
            (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
        {
            pRuntimeApiTrace = pRuntimeApiTrace + KERNEL;
        }
        else if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)
        {
            pRuntimeApiTrace = pRuntimeApiTrace + THREAD_SYNC;
        }
        else if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
        {
            pRuntimeApiTrace = pRuntimeApiTrace + MEMCPY_H2D1 + s_MemTransCount;
        }

        if (pCallbackData->callbackSite == CUPTI_API_ENTER)
        {
            // For a kernel launch report the kernel name.
            // Otherwise use the API function name.
            if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
                callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)
            {
                pRuntimeApiTrace->pFunctionName = pCallbackData->symbolName;
            }
            else
            {
                pRuntimeApiTrace->pFunctionName = pCallbackData->functionName;
            }

            // Store parameters passed to cudaMemcpy.
            if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
            {
                pRuntimeApiTrace->memcpyBytes = ((cudaMemcpy_v3020_params *)(pCallbackData->functionParams))->count;
                pRuntimeApiTrace->memcpyKind = ((cudaMemcpy_v3020_params *)(pCallbackData->functionParams))->kind;
            }

            // Collect timestamp for API start.
            CUPTI_API_CALL(cuptiGetTimestamp(&startTimestamp));

            pRuntimeApiTrace->startTimestamp = startTimestamp;
        }

        if (pCallbackData->callbackSite == CUPTI_API_EXIT)
        {
            // Collect timestamp for API exit.
            CUPTI_API_CALL(cuptiGetTimestamp(&endTimestamp));

            pRuntimeApiTrace->endTimestamp = endTimestamp;

            // Advance to the next memory transfer operation.
            if (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
            {
                s_MemTransCount++;
            }
        }
    }
}

static const char *
GetMemcpyKind(
    enum cudaMemcpyKind memcpyKind)
{
    switch (memcpyKind)
    {
        case cudaMemcpyHostToDevice:
            return "HostToDevice";
        case cudaMemcpyDeviceToHost:
            return "DeviceToHost";
        default:
            return "<unknown>";
    }
}

static void
DisplayTimestamps(
    RuntimeApiTrace *pRuntimeApiTrace)
{
    // Calculate timestamp of kernel based on timestamp from cudaDeviceSynchronize() call.
    pRuntimeApiTrace[KERNEL].endTimestamp = pRuntimeApiTrace[THREAD_SYNC].endTimestamp;

    printf("Start timestamp or duration reported in nanoseconds.\n\n");
    printf("Name\t\t\tStart Time\t\tDuration\tBytes\tKind\n");

    printf("%s\t\t%llu\t%llu\t\t%llu\t%s\n", pRuntimeApiTrace[MEMCPY_H2D1].pFunctionName,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D1].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D1].endTimestamp - pRuntimeApiTrace[MEMCPY_H2D1].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D1].memcpyBytes,
            GetMemcpyKind(pRuntimeApiTrace[MEMCPY_H2D1].memcpyKind));

    printf("%s\t\t%llu\t%llu\t\t%llu\t%s\n", pRuntimeApiTrace[MEMCPY_H2D2].pFunctionName,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D2].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D2].endTimestamp - pRuntimeApiTrace[MEMCPY_H2D2].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_H2D2].memcpyBytes,
            GetMemcpyKind(pRuntimeApiTrace[MEMCPY_H2D2].memcpyKind));

    printf("%s\t\t%llu\t%llu\t\t%llu\t%s\n", pRuntimeApiTrace[MEMCPY_D2H].pFunctionName,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_D2H].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_D2H].endTimestamp - pRuntimeApiTrace[MEMCPY_D2H].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[MEMCPY_D2H].memcpyBytes,
            GetMemcpyKind(pRuntimeApiTrace[MEMCPY_D2H].memcpyKind));

    printf("%s\t%llu\t%llu\t\tNA\tNA\n", pRuntimeApiTrace[KERNEL].pFunctionName,
            (unsigned long long)pRuntimeApiTrace[KERNEL].startTimestamp,
            (unsigned long long)pRuntimeApiTrace[KERNEL].endTimestamp - pRuntimeApiTrace[KERNEL].startTimestamp);
}

static void
CleanUp(
    int *pHostA,
    int *pHostB,
    int *pHostC,
    int *pDeviceA,
    int *pDeviceB,
    int *pDeviceC)
{
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

int
main(
    int argc,
    char *argv[])
{
    CUcontext context = 0;
    CUdevice device = 0;

    int N = 50000;
    size_t size = N * sizeof(int);

    int threadsPerBlock = 0;
    int blocksPerGrid = 0;

    int sum, i;

    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;

    CUpti_SubscriberHandle subscriber;
    RuntimeApiTrace pRuntimeApiTrace[LAUNCH_LAST];

    // Subscribe to CUPTI callbacks.
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)TimestampCallback , &pRuntimeApiTrace));

    // Enable all callbacks for CUDA Runtime APIs.
    // Callback will be invoked at the entry and exit points of each of the CUDA Runtime API.
    CUPTI_API_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

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

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        sum = pHostA[i] + pHostB[i];
        if (pHostC[i] != sum)
        {
            printf("Error: Kernel execution failed.\n");
            goto Error;
        }
    }

    // Display timestamps collected in the callback.
    DisplayTimestamps(pRuntimeApiTrace);

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    CUPTI_API_CALL(cuptiUnsubscribe(subscriber));

    exit(EXIT_SUCCESS);

Error:
    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    exit(EXIT_FAILURE);
}

