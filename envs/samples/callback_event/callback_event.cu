/*
 * Copyright 2010-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler event values
 * using callbacks for CUDA runtime APIs
 *
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string>

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include <cupti.h>
#include "helper_cupti.h"

// Macros
#define EVENT_NAME "inst_executed"

// Structures
typedef struct CuptiEventData_st
{
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
} CuptiEventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st
{
    CuptiEventData *pEventData;
    uint64_t eventValue;
} RuntimeApiTrace;

// Kernels
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
GetEventValueCallback(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const CUpti_CallbackData *pCallbackInfo)
{
    RuntimeApiTrace *pRuntimeApiTrace = (RuntimeApiTrace *)pUserData;
    size_t bytesRead;

    // This callback is enabled only for launch so we shouldn't see anything else.
    if ((callbackId != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
        (callbackId != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        printf("Error: %s:%d: Unexpected CBID: %d\n", __FILE__, __LINE__, callbackId);
        exit(EXIT_FAILURE);
    }

    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
    {
        cudaDeviceSynchronize();
        CUPTI_API_CALL(cuptiSetEventCollectionMode(pCallbackInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
        CUPTI_API_CALL(cuptiEventGroupEnable(pRuntimeApiTrace->pEventData->eventGroup));
    }

    if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
    {
        uint32_t numInstances = 0, i;
        uint64_t *pValues = NULL;
        size_t valueSize = sizeof(numInstances);

        CUPTI_API_CALL(cuptiEventGroupGetAttribute(pRuntimeApiTrace->pEventData->eventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &valueSize, &numInstances));

        bytesRead = sizeof (uint64_t) * numInstances;
        pValues = (uint64_t *)malloc(bytesRead);
        MEMORY_ALLOCATION_CALL(pValues);

        cudaDeviceSynchronize();

        CUPTI_API_CALL(cuptiEventGroupReadEvent(pRuntimeApiTrace->pEventData->eventGroup, CUPTI_EVENT_READ_FLAG_NONE, pRuntimeApiTrace->pEventData->eventId, &bytesRead, pValues));

        pRuntimeApiTrace->eventValue = 0;
        for (i=0; i<numInstances; i++)
        {
            pRuntimeApiTrace->eventValue += pValues[i];
        }

        free(pValues);

        CUPTI_API_CALL(cuptiEventGroupDisable(pRuntimeApiTrace->pEventData->eventGroup));
    }
}

static void
DisplayEventValue(
    RuntimeApiTrace *pRuntimeApiTrace,
    const char *pEventName)
{
    printf("Event Name : %s \n", pEventName);
    printf("Event Value : %llu\n", (unsigned long long) pRuntimeApiTrace->eventValue);
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
        cudaFree(pDeviceA);
    }
    if (pDeviceB)
    {
        cudaFree(pDeviceB);
    }
    if (pDeviceC)
    {
        cudaFree(pDeviceC);
    }
}

int
main(
    int argc,
    char *argv[])
{
    CUcontext context = 0;
    CUdevice dev = 0;
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int sum, i;
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor=0;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    int deviceNum;
    int deviceCount;
    char deviceName[256];
    const char *pEventName;
    uint32_t profileAll = 1;

    CUpti_SubscriberHandle subscriber;
    CuptiEventData cuptiEvent;
    RuntimeApiTrace runtimeApiTrace;

    printf("Usage: %s [device_num] [event_name]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    if (argc > 1)
    {
        deviceNum = atoi(argv[1]);
    }
    else
    {
        deviceNum = 0;
    }
    printf("CUDA Device Number: %d\n", deviceNum);

    DRIVER_API_CALL(cuDeviceGet(&dev, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, dev));
    printf("CUDA Device Name: %s\n", deviceName);

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);
    int deviceComputeCapability = 10 * computeCapabilityMajor + computeCapabilityMinor;
    if (deviceComputeCapability > 72)
    {
        printf("Warning: Sample unsupported on Device with compute capability > 7.2. Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    DRIVER_API_CALL(cuCtxCreate(&context, 0, dev));

    // Creating event group for profiling
    CUPTI_API_CALL(cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0));

    if (argc > 2)
    {
        pEventName = argv[2];
    }
    else
    {
        pEventName = EVENT_NAME;
    }

    CUPTI_API_CALL(cuptiEventGetIdFromName(dev, pEventName, &cuptiEvent.eventId));
    CUPTI_API_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId));
    CUPTI_API_CALL(cuptiEventGroupSetAttribute(cuptiEvent.eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profileAll), &profileAll));

    runtimeApiTrace.pEventData = &cuptiEvent;

    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)GetEventValueCallback, &runtimeApiTrace));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // Allocate input vectors pHostA and pHostB in host memory
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

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&pDeviceC, size));

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());

    // Copy result from device memory to host memory
    // pHostC contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        sum = pHostA[i] + pHostB[i];
        if (pHostC[i] != sum)
        {
            printf("kernel execution FAILED\n");
            goto Error;
        }
    }

    DisplayEventValue(&runtimeApiTrace, pEventName);

    runtimeApiTrace.pEventData = NULL;

    CUPTI_API_CALL(cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId));
    CUPTI_API_CALL(cuptiEventGroupDestroy(cuptiEvent.eventGroup));
    CUPTI_API_CALL(cuptiUnsubscribe(subscriber));

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    exit(EXIT_SUCCESS);

Error:
    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    exit(EXIT_FAILURE);
}

