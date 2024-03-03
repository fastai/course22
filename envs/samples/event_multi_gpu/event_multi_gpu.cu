/*
* Copyright 2014-2017 NVIDIA Corporation. All rights reserved.
*
* Sample app to demonstrate use of CUPTI library to obtain profiler
* event values on a multi-gpu setup without serializing the compute
* work on the devices.
*/

// System headers
#include <stdio.h>
#include <stdlib.h>

// CUDA headers
#include <cuda_runtime_api.h>

// CUPTI headers
#include <cupti_events.h>
#include "helper_cupti.h"

// Macros
#define MAX_DEVICES     32
#define BLOCK_X         32
#define GRID_X          32

// Default events.
#define EVENT_NAME      "inst_executed"

// Kernels
__global__ void
Kernel()
{
    uint64_t i = 0;
    volatile uint64_t limit = 1024 * 128;

    for (i = 0; i < limit; i++)
    {
    }
}

// Functions
int
main(
    int argc,
    char *argv[])
{
    CUdevice device[MAX_DEVICES];
    CUcontext context[MAX_DEVICES];

    int i = 0, j = 0;
    int deviceCount;
    char deviceName[256];
    size_t bytesRead, valueSize;
    uint32_t numInstances = 0;
    uint32_t profileAll = 1;

    CUpti_EventGroup eventGroup[MAX_DEVICES];
    CUpti_EventID eventId[MAX_DEVICES];
    const char *pEventName;
    uint64_t *pEventValues = NULL, eventValue = 0;

    printf("Usage: %s [event_name]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    if (deviceCount < 2)
    {
        printf("Warning: This multi-GPU test is waived on single GPU setup.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    if (deviceCount > MAX_DEVICES)
    {
        printf("Warning: Found more devices (%d) than handled in the test (%d).\nWaiving test.\n", deviceCount, MAX_DEVICES);
        exit(EXIT_WAIVED);
    }

    if (argc > 1)
    {
        pEventName = argv[1];
    }
    else
    {
        pEventName = EVENT_NAME;
    }

    for (i = 0; i < deviceCount; i++)
    {
        DRIVER_API_CALL(cuDeviceGet(&device[i], i));
        DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device[i]));

        printf("CUDA Device Name: %s\n", deviceName);
    }

    // Create one context per device.
    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxCreate(&(context[i]), 0, device[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&(context[i])));
    }

    // Enable event profiling on each device.
    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        CUPTI_API_CALL(cuptiSetEventCollectionMode(context[i], CUPTI_EVENT_COLLECTION_MODE_KERNEL));
        CUPTI_API_CALL(cuptiEventGroupCreate(context[i], &eventGroup[i], 0));

        CUPTI_API_CALL(cuptiEventGetIdFromName(device[i], pEventName, &eventId[i]));

        CUPTI_API_CALL(cuptiEventGroupAddEvent(eventGroup[i], eventId[i]));
        CUPTI_API_CALL(cuptiEventGroupSetAttribute(eventGroup[i], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profileAll), &profileAll));
        CUPTI_API_CALL(cuptiEventGroupEnable(eventGroup[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // Launch kernel on each device.
    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        Kernel<<<GRID_X, BLOCK_X>>>();

        // Don't do any sync here, it's done once.
        // Work is queued on all devices.

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // Sync each context now.
    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        DRIVER_API_CALL(cuCtxSynchronize());

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    // Read events.
    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaSetDevice(i));

        DRIVER_API_CALL(cuCtxPushCurrent(context[i]));

        valueSize = sizeof(numInstances);
        CUPTI_API_CALL(cuptiEventGroupGetAttribute(eventGroup[i], CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &valueSize, &numInstances));

        bytesRead = sizeof(uint64_t) * numInstances;
        pEventValues = (uint64_t *)malloc(bytesRead);
        MEMORY_ALLOCATION_CALL(pEventValues);

        CUPTI_API_CALL(cuptiEventGroupReadEvent(eventGroup[i], CUPTI_EVENT_READ_FLAG_NONE, eventId[i], &bytesRead, pEventValues));

        if (bytesRead != (sizeof(uint64_t) * numInstances))
        {
            printf("Error: Failed to read value for \"%s\"\n", pEventName);
            exit(EXIT_FAILURE);
        }

        for (j = 0; j < numInstances; j++)
        {
            eventValue += pEventValues[j];
        }

        printf("[%d] %s: %llu\n", i, pEventName, (unsigned long long)eventValue);

        CUPTI_API_CALL(cuptiEventGroupDisable(eventGroup[i]));
        CUPTI_API_CALL(cuptiEventGroupDestroy(eventGroup[i]));

        DRIVER_API_CALL(cuCtxPopCurrent(&context[i]));
    }

    exit(EXIT_SUCCESS);
}
