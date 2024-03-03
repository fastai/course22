/*
 * Copyright 2011-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain metric values
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
#include "helper_cupti_activity.h"

// Macros
#define METRIC_NAME "ipc"

// Variables

// User data for event collection callback.
typedef struct MetricData_st
{
    // The device where metric is being collected.
    CUdevice device;
    // The set of event groups to collect for a pass.
    CUpti_EventGroupSet *pEventGroups;
    // The current number of events collected in pEventIdArray and pEventValueArray.
    uint32_t eventIdx;
    // The number of entries in pEventIdArray and pEventValueArray.
    uint32_t numEvents;
    // Array of event ids.
    CUpti_EventID *pEventIdArray;
    // Array of event values.
    uint64_t *pEventValueArray;
} MetricData;

static uint64_t s_KernelDuration;

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

// Functions
static void
InitializeVector(
    int *pVector,
    int N)
{
    for (int i = 0; i <  N; i++)
    {
        pVector[i] = i;
    }
}

static void
DoPass()
{
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    int i, sum;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors.
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

    // Invoke kernel.
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result.
    for (i = 0; i < N; ++i)
    {
        sum = pHostA[i] + pHostB[i];
        if (pHostC[i] != sum)
        {
            fprintf(stderr, "Error: Result verification failed.\n");
            exit(EXIT_FAILURE);
        }
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

void CUPTIAPI
MetricCallbackHandler(
    void *pUserdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const CUpti_CallbackData *pCallbackData)
{
    MetricData *pMetricData = (MetricData*)pUserdata;
    unsigned int i, j, k;

    // This callback is enabled only for launch so we shouldn't see anything else.
    if ((callbackId != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
        (callbackId != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        printf("Error: %s:%d: Unexpected Callback Id: %d.\n", __FILE__, __LINE__, callbackId);
        exit(EXIT_FAILURE);
    }

    // On entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event.
    if (pCallbackData->callbackSite == CUPTI_API_ENTER)
    {
        RUNTIME_API_CALL(cudaDeviceSynchronize());

        CUPTI_API_CALL(cuptiSetEventCollectionMode(pCallbackData->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

        for (i = 0; i < pMetricData->pEventGroups->numEventGroups; i++)
        {
            uint32_t all = 1;
            CUPTI_API_CALL(cuptiEventGroupSetAttribute(pMetricData->pEventGroups->eventGroups[i], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all));
            CUPTI_API_CALL(cuptiEventGroupEnable(pMetricData->pEventGroups->eventGroups[i]));
        }
    }

    // On exit, read and record event values.
    if (pCallbackData->callbackSite == CUPTI_API_EXIT)
    {
        RUNTIME_API_CALL(cudaDeviceSynchronize());

        // For each group, read the event values from the group and record in pMetricData.
        for (i = 0; i < pMetricData->pEventGroups->numEventGroups; i++)
        {
            CUpti_EventGroup group = pMetricData->pEventGroups->eventGroups[i];
            CUpti_EventDomainID groupDomain;
            CUpti_EventID *pEventIds;

            uint32_t numEvents, numInstances, numTotalInstances;

            size_t groupDomainSize = sizeof(groupDomain);
            size_t numEventsSize = sizeof(numEvents);
            size_t numInstancesSize = sizeof(numInstances);
            size_t numTotalInstancesSize = sizeof(numTotalInstances);
            size_t valuesSize, eventIdsSize;
            size_t numCountersRead = 0;

            uint64_t *pValues, normalized, *pSum;

            CUPTI_API_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain));
            CUPTI_API_CALL(cuptiDeviceGetEventDomainAttribute(pMetricData->device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize, &numTotalInstances));
            CUPTI_API_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstancesSize, &numInstances));
            CUPTI_API_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));

            eventIdsSize = numEvents * sizeof(CUpti_EventID);
            pEventIds = (CUpti_EventID *)malloc(eventIdsSize);
            MEMORY_ALLOCATION_CALL(pEventIds);

            CUPTI_API_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, pEventIds));

            valuesSize = sizeof(uint64_t) * numInstances * numEvents;
            pValues = (uint64_t *)malloc(valuesSize);
            MEMORY_ALLOCATION_CALL(pValues);

            CUPTI_API_CALL(cuptiEventGroupReadAllEvents(group, CUPTI_EVENT_READ_FLAG_NONE, &valuesSize, pValues, &eventIdsSize, pEventIds, &numCountersRead));

            if (pMetricData->eventIdx >= pMetricData->numEvents)
            {
                fprintf(stderr, "Error: Too many events collected, metric expects only %d events.\n", (int)pMetricData->numEvents);
                exit(EXIT_FAILURE);
            }

            pSum = (uint64_t *)calloc(sizeof(uint64_t), numEvents);
            // Sum collected of event values from all instances.
            for (k = 0; k < numInstances; k++)
            {
                for (j = 0; j < numEvents; j++)
                {
                    pSum[j] += pValues[(k * numEvents) + j];
                }
            }

            for (j = 0; j < numEvents; j++)
            {
                // Normalize the event value to represent the total number of domain instances on the device.
                normalized = (pSum[j] * numTotalInstances) / numInstances;

                pMetricData->pEventIdArray[pMetricData->eventIdx] = pEventIds[j];
                pMetricData->pEventValueArray[pMetricData->eventIdx] = normalized;
                pMetricData->eventIdx++;

                // Print collected value.
                {
                    char eventName[128];
                    size_t eventNameSize = sizeof(eventName) - 1;
                    CUPTI_API_CALL(cuptiEventGetAttribute(pEventIds[j], CUPTI_EVENT_ATTR_NAME,
                                                      &eventNameSize, eventName));

                    eventName[127] = '\0';
                    printf("\t%s = %llu (", eventName, (unsigned long long)pSum[j]);
                    if (numInstances > 1)
                    {
                        for (k = 0; k < numInstances; k++)
                        {
                            if (k != 0)
                            {
                                printf(", ");
                            }
                            printf("%llu", (unsigned long long)pValues[(k * numEvents) + j]);
                        }
                    }

                    printf(")\n");
                    printf("\t%s (normalized) (%llu * %u) / %u = %llu\n",
                            eventName, (unsigned long long)pSum[j],
                            numTotalInstances, numInstances,
                            (unsigned long long)normalized);
                }
            }

            free(pValues);
            free(pSum);
        }

        for (i = 0; i < pMetricData->pEventGroups->numEventGroups; i++)
        {
            CUPTI_API_CALL(cuptiEventGroupDisable(pMetricData->pEventGroups->eventGroups[i]));
        }
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
    pUserData->skipCuptiSubscription       = 1;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable activity record kinds.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
}

int
main(
    int argc,
    char *argv[])
{
    CUcontext context = 0;
    CUdevice device = 0;

    int deviceNum;
    int deviceCount;
    char deviceName[256];

    const char *pMetricName;
    CUpti_MetricID metricId;
    CUpti_EventGroupSets *pPassData;
    CUpti_MetricValue metricValue;

    MetricData metricData;
    unsigned int pass;

    printf("Usage: %s [device_num] [metric_name].\n", argv[0]);

    // Initialize CUDA
    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\n Waiving sample.\n");
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

    DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
    printf("CUDA Device Name: %s\n", deviceName);

    int major, minor;
    DRIVER_API_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    DRIVER_API_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    int deviceComputeCapability = 10 * major + minor;
    printf("Compute Capability: %d.%d\n", major, minor);
    if (deviceComputeCapability > 72)
    {
        printf("Warning: Sample unsupported on Device with compute capability > 7.2.\n Waiving sample.\n");
        exit(EXIT_WAIVED);
    }

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

    // Get the name of the metric to collect
    if (argc > 2)
    {
        pMetricName = argv[2];
    }
    else
    {
        pMetricName = METRIC_NAME;
    }

    // Need to collect duration of kernel execution without any event
    // collection enabled (some metrics need kernel duration as part of
    // calculation). The only accurate way to do this is by using the
    // activity API.
    {
        SetupCupti();
        DoPass();
        RUNTIME_API_CALL(cudaDeviceSynchronize());
        CUPTI_API_CALL(cuptiActivityFlushAll(0));
    }

    // Subscribe to CUPTI callbacks with metric data.
    CUPTI_API_CALL(cuptiSubscribe(&globals.subscriberHandle, (CUpti_CallbackFunc)MetricCallbackHandler, &metricData));
    CUPTI_API_CALL(cuptiEnableCallback(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_API_CALL(cuptiEnableCallback(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // Allocate space to hold all the events needed for the metric.
    CUPTI_API_CALL(cuptiMetricGetIdFromName(device, pMetricName, &metricId));
    CUPTI_API_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
    metricData.device = device;
    metricData.pEventIdArray = (CUpti_EventID *)malloc(metricData.numEvents * sizeof(CUpti_EventID));
    MEMORY_ALLOCATION_CALL(metricData.pEventIdArray);
    metricData.pEventValueArray = (uint64_t *)malloc(metricData.numEvents * sizeof(uint64_t));
    MEMORY_ALLOCATION_CALL(metricData.pEventValueArray);
    metricData.eventIdx = 0;

    // Get the number of passes required to collect all the events.
    // Needed for the metric and the event groups for each pass.
    CUPTI_API_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId, &pPassData));
    for (pass = 0; pass < pPassData->numSets; pass++)
    {
        printf("Pass: %u\n", pass);
        metricData.pEventGroups = pPassData->sets + pass;
        DoPass();
    }

    if (metricData.eventIdx != metricData.numEvents)
    {
        fprintf(stderr, "Error: Expected %u metric events, got %u metric events.\n", metricData.numEvents, metricData.eventIdx);
        exit(EXIT_FAILURE);
    }

    // Use all the collected events to calculate the metric value.
    CUPTI_API_CALL(cuptiMetricGetValue(device, metricId, metricData.numEvents * sizeof(CUpti_EventID), metricData.pEventIdArray, metricData.numEvents * sizeof(uint64_t), metricData.pEventValueArray, s_KernelDuration, &metricValue));

    // Print the metric value, we format based on the value kind.
    {
        CUpti_MetricValueKind valueKind;
        size_t valueKindSize = sizeof(valueKind);

        CUPTI_API_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
        switch (valueKind)
        {
            case CUPTI_METRIC_VALUE_KIND_DOUBLE:
                printf("Metric %s = %f\n", pMetricName, metricValue.metricValueDouble);
                break;
            case CUPTI_METRIC_VALUE_KIND_UINT64:
                printf("Metric %s = %llu\n", pMetricName, (unsigned long long)metricValue.metricValueUint64);
                break;
            case CUPTI_METRIC_VALUE_KIND_INT64:
                printf("Metric %s = %lld\n", pMetricName, (long long)metricValue.metricValueInt64);
                break;
            case CUPTI_METRIC_VALUE_KIND_PERCENT:
                printf("Metric %s = %f%%\n", pMetricName, metricValue.metricValuePercent);
                break;
            case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
                printf("Metric %s = %llu bytes/sec\n", pMetricName, (unsigned long long)metricValue.metricValueThroughput);
                break;
            case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
                printf("Metric %s = utilization level %u\n", pMetricName, (unsigned int)metricValue.metricValueUtilizationLevel);
                break;
            default:
                fprintf(stderr, "Error: unknown value kind.\n");
                exit(EXIT_FAILURE);
        }
    }

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
