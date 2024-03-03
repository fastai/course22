/*
* Copyright 2015-2022 NVIDIA Corporation. All rights reserved.
*
* Sample to demonstrate use of NVlink CUPTI APIs
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
#include "helper_cupti_activity.h"

#define MAX_DEVICES    (32)
#define BLOCK_SIZE     (1024)
#define GRID_SIZE      (512)
#define NUM_METRIC     (4)
#define NUM_EVENTS     (2)
#define MAX_SIZE       (64*1024*1024)
#define NUM_STREAMS    (6)

int cpuToGpu       = 0;
int gpuToGpu       = 0;
int cpuToGpuAccess = 0;
int gpuToGpuAccess = 0;
bool metricSupport = true;

CUpti_ActivityNvLink4 *nvlinkRec = NULL;

extern "C" __global__ void
TestNvLinkBandwidth(
    float *pSource,
    float *pDestination)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    pDestination[idx] = pSource[idx] * 2.0f;
}

#define DIM(x) (sizeof(x)/sizeof(*(x)))

void CalculateSize(
    char *pResult,
    uint64_t size)
{
    int i;

    const char *pSizes[]   = { "TB", "GB", "MB", "KB", "B" };
    uint64_t  exbiBytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;

    uint64_t  multiplier = exbiBytes;

    for (i = 0; (unsigned)i < DIM(pSizes); i++, multiplier /= (uint64_t)1024)
    {
        if (size < multiplier)
        {
            continue;
        }

        sprintf(pResult, "%.1f %s", (float) size / multiplier, pSizes[i]);

        return;
    }

    strcpy(pResult, "0");

    return;
}

void
ReadMetricValue(
    CUpti_EventGroup eventGroup,
    uint32_t numEvents,
    CUdevice device,
    CUpti_MetricID *pMetricId,
    uint64_t timeDuration,
    CUpti_MetricValue *pMetricValue)
{

    size_t bufferSizeBytes, numCountersRead;
    size_t arraySizeBytes = 0;
    size_t numTotalInstancesSize = 0;
    size_t domainSize;
    uint64_t numTotalInstances = 0;

    uint64_t *pAggrEventValueArray = NULL;
    size_t aggrEventValueArraySize;

    uint64_t *pEventValueArray = NULL;

    CUpti_EventID *pEventIdArray;
    CUpti_EventDomainID domainId;

    uint32_t i = 0, j = 0;
    domainSize = sizeof(CUpti_EventDomainID);

    CUPTI_API_CALL(cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &domainSize, (void *)&domainId));

    numTotalInstancesSize = sizeof(uint64_t);

    CUPTI_API_CALL(cuptiDeviceGetEventDomainAttribute(device, domainId, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize, (void *)&numTotalInstances));

    arraySizeBytes = sizeof(CUpti_EventID) * numEvents;
    bufferSizeBytes = sizeof(uint64_t) * numEvents * numTotalInstances;

    pEventValueArray = (uint64_t *)malloc(bufferSizeBytes);
    MEMORY_ALLOCATION_CALL(pEventValueArray);

    pEventIdArray = (CUpti_EventID *)malloc(arraySizeBytes);
    MEMORY_ALLOCATION_CALL(pEventIdArray);

    pAggrEventValueArray = (uint64_t *)calloc(numEvents, sizeof(uint64_t));
    MEMORY_ALLOCATION_CALL(pAggrEventValueArray);

    aggrEventValueArraySize = sizeof(uint64_t) * numEvents;

    CUPTI_API_CALL(cuptiEventGroupReadAllEvents(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, &bufferSizeBytes, pEventValueArray, &arraySizeBytes, pEventIdArray, &numCountersRead));

    for (i = 0; i < numEvents; i++)
    {
        for (j = 0; j < numTotalInstances; j++)
        {
            pAggrEventValueArray[i] += pEventValueArray[i + numEvents * j];
        }
    }

    for (i = 0; i < NUM_METRIC; i++)
    {
        CUPTI_API_CALL(cuptiMetricGetValue(device, pMetricId[i], arraySizeBytes, pEventIdArray, aggrEventValueArraySize, pAggrEventValueArray, timeDuration, &pMetricValue[i]));
    }

    free(pEventValueArray);
    free(pEventIdArray);
}

// Print metric value, we format based on the value kind.
int
PrintMetricValue(
    CUpti_MetricID metricId,
    CUpti_MetricValue metricValue,
    const char *pMetricName)
{

    CUpti_MetricValueKind valueKind;
    char stringArray[64];
    size_t valueKindSize = sizeof(valueKind);

    CUPTI_API_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
    switch (valueKind)
    {

        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            printf("%s = ", pMetricName);
            CalculateSize(stringArray, (uint64_t)metricValue.metricValueDouble);
            printf("%s\n", stringArray);
            break;

        case CUPTI_METRIC_VALUE_KIND_UINT64:
            printf("%s = ", pMetricName);
            CalculateSize(stringArray, (uint64_t)metricValue.metricValueUint64);
            printf("%s\n", stringArray);
            break;

        case CUPTI_METRIC_VALUE_KIND_INT64:
            printf("%s = ", pMetricName);
            CalculateSize(stringArray, (uint64_t)metricValue.metricValueInt64);
            printf("%s\n", stringArray);
            break;

        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            printf("%s = ", pMetricName);
            CalculateSize(stringArray, (uint64_t)metricValue.metricValueThroughput);
            printf("%s/Sec\n", stringArray);
            break;

        default:
            fprintf(stderr, "Error: Unknown value kind.\n");
            return -1;
    }

    return 0;
  }

void
TestCpuToGpu(
    CUpti_EventGroup *eventGroup,
    CUdeviceptr *pDevBuffer,
    float **pHostBuffer,
    size_t bufferSize,
    cudaStream_t *pCudaStreams,
    uint64_t *pTimeDuration,
    int numEventGroup)
{
    int i;
    uint32_t value = 1;
    uint64_t startTimestamp, endTimestamp;

    for (i = 0; i < numEventGroup; i++)
    {
        CUPTI_API_CALL(cuptiEventGroupEnable(eventGroup[i]));
        CUPTI_API_CALL(cuptiEventGroupSetAttribute(eventGroup[i], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(uint32_t), (void*)&value));
    }

    CUPTI_API_CALL(cuptiGetTimestamp(&startTimestamp));

    // Unidirectional copy H2D.
    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Unidirectional copy D2H.
    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i], (void *)pDevBuffer[i], bufferSize, cudaMemcpyDeviceToHost, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Bidirectional copy.
    for (i = 0; i < NUM_STREAMS; i += 2)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, pCudaStreams[i]));
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i + 1], (void *)pDevBuffer[i + 1], bufferSize, cudaMemcpyDeviceToHost, pCudaStreams[i + 1]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    CUPTI_API_CALL(cuptiGetTimestamp(&endTimestamp));
    *pTimeDuration = endTimestamp - startTimestamp;
}

void
TestGpuToGpu(
    CUpti_EventGroup *pEventGroup,
    CUdeviceptr *pDevBufferA,
    CUdeviceptr *pDevBufferB,
    float** pHostBuffer,
    size_t bufferSize,
    cudaStream_t *pCudaStreams,
    uint64_t *pTimeDuration,
    int numEventGroup)
{
    int i;
    uint32_t value = 1;
    uint64_t startTimestamp, endTimestamp;

    for (i = 0; i < numEventGroup; i++)
    {
        CUPTI_API_CALL(cuptiEventGroupEnable(pEventGroup[i]));
        CUPTI_API_CALL(cuptiEventGroupSetAttribute(pEventGroup[i], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(uint32_t), (void*)&value));
    }

    RUNTIME_API_CALL(cudaSetDevice(0));
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(1, 0));
    RUNTIME_API_CALL(cudaSetDevice(1));
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(0, 0));

    // Unidirectional copy H2D.
    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBufferA[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBufferB[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    CUPTI_API_CALL(cuptiGetTimestamp(&startTimestamp));

    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBufferA[i], (void *)pDevBufferB[i], bufferSize, cudaMemcpyDeviceToDevice, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBufferB[i], (void *)pDevBufferA[i], bufferSize, cudaMemcpyDeviceToDevice, pCudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++)
    {
        TestNvLinkBandwidth <<< GRID_SIZE, BLOCK_SIZE >>> ((float*)pDevBufferB[i], (float*)pDevBufferA[i]);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    CUPTI_API_CALL(cuptiGetTimestamp(&endTimestamp));
    *pTimeDuration = endTimestamp - startTimestamp;
}

void
NVLinkRecords(
    CUpti_Activity *pRecord)
{
    switch(pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_NVLINK:
        {
            nvlinkRec = (CUpti_ActivityNvLink4 *)pRecord;
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
    pUserData->pPostProcessActivityRecords = NVLinkRecords;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, NULL, stdout);

    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NVLINK));
}

static void
PrintUsage()
{
    printf("Usage: Demonstrate the use of NVLink CUPTI APIs.\n");
    printf("       --help           : Display help message.\n");
    printf("       --cpu-to-gpu    : Show results for data transfer between CPU and GPU.\n");
    printf("       --gpu-to-gpu    : Show results for data transfer between two GPUs.\n");
}

void
ParseCommandLineArgs(
    int argc,
    char *argv[])
{
    if (argc != 2)
    {
        printf("Error: Invalid number of options.\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "--cpu-to-gpu") == 0)
    {
        cpuToGpu = 1;
    }
    else if (strcmp(argv[1], "--gpu-to-gpu") == 0)
    {
        gpuToGpu = 1;
    }
    else if ((strcmp(argv[1], "--help") == 0) || (strcmp(argv[1], "-help") == 0) || (strcmp(argv[1], "-h") == 0))
    {
        PrintUsage();
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("Error: Invalid/Incomplete option: %s.\n", argv[1]);
        exit(EXIT_FAILURE);
    }
}

int
main(
    int argc,
    char *argv[])
{
    int deviceCount = 0, i = 0, j = 0, numEventGroup = 0;
    size_t bufferSize = 0, freeMemory = 0, totalMemory = 0;
    CUpti_EventGroupSets *pPasses = NULL;
    CUcontext context;
    char stringArray[64];

    CUdeviceptr devBufferA[NUM_STREAMS];
    CUdeviceptr devBufferB[NUM_STREAMS];
    float* pHostBuffer[NUM_STREAMS];

    cudaStream_t cudaStreams[NUM_STREAMS] = {0};

    CUpti_EventGroup eventGroup[32];
    CUpti_MetricID metricId[NUM_METRIC];
    uint32_t numEvents[NUM_METRIC];
    CUpti_MetricValue metricValue[NUM_METRIC];
    cudaDeviceProp deviceProp[MAX_DEVICES];
    uint64_t timeDuration;

    // Adding NVLink Metrics.
    const char *pMetricName[NUM_METRIC] =   {
                                                "nvlink_total_data_transmitted",
                                                "nvlink_total_data_received",
                                                "nvlink_transmit_throughput",
                                                "nvlink_receive_throughput"
                                            };

    // Parse command line arguments.
    ParseCommandLineArgs(argc, argv);

    SetupCupti();

    // Initialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));

    printf("There are %d devices.\n", deviceCount);

    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving.\n");
        exit(EXIT_WAIVED);
    }

    for (i = 0; i < deviceCount; i++)
    {
        RUNTIME_API_CALL(cudaGetDeviceProperties(&deviceProp[i], i));
        printf("CUDA Device %d Name: %s\n", i, deviceProp[i].name);

        // Check if any device is Turing+.
        if (deviceProp[i].major == 7 && deviceProp[i].minor > 0)
        {
            metricSupport = false;
        }
        else if (deviceProp[i].major > 7)
        {
            metricSupport = false;
        }
    }

    // Set memcpy size based on available device memory.
    RUNTIME_API_CALL(cudaMemGetInfo(&freeMemory, &totalMemory));
    bufferSize = MAX_SIZE < (freeMemory/4) ? MAX_SIZE : (freeMemory/4);

    printf("Total Device Memory available: ");
    CalculateSize(stringArray, (uint64_t)totalMemory);
    printf("%s\n", stringArray);

    printf("Memcpy size set to:%llu bytes (%llu MB)\n", (unsigned long long)bufferSize, (unsigned long long)bufferSize/(1024*1024));

    for(i = 0; i < NUM_STREAMS; i++)
    {
       RUNTIME_API_CALL(cudaStreamCreate(&cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Nvlink-topology Records are generated even before cudaMemcpy API is called.
    CUPTI_API_CALL(cuptiActivityFlushAll(0));

    // Transfer Data between Host And Device, if Nvlink is Present.
    // Check condition : nvlinkRec->flag & CUPTI_LINK_FLAG_SYSMEM_ACCESS.
    // True : Nvlink is present between CPU & GPU.
    // False : Nvlink is not present.
    if ((nvlinkRec) && (((cpuToGpu) && (cpuToGpuAccess)) || ((gpuToGpu) && (gpuToGpuAccess))))
    {
        if (!metricSupport)
        {
            printf("Warning: Legacy CUPTI metrics not supported from Turing+ devices.\nWaiving.\n");
            exit(EXIT_WAIVED);
        }

        for (i = 0; i < NUM_METRIC; i++)
        {
            CUPTI_API_CALL(cuptiMetricGetIdFromName(0, pMetricName[i], &metricId[i]));
            CUPTI_API_CALL(cuptiMetricGetNumEvents(metricId[i], &numEvents[i]));
        }

        DRIVER_API_CALL(cuCtxCreate(&context, 0, 0));

        CUPTI_API_CALL(cuptiMetricCreateEventGroupSets(context, (sizeof metricId) ,metricId, &pPasses));

        // EventGroups required to profile Nvlink metrics.
        for (i = 0; i < (signed)pPasses->numSets; i++)
        {
            for (j = 0; j < (signed)pPasses->sets[i].numEventGroups; j++)
            {
                eventGroup[numEventGroup] = pPasses->sets[i].eventGroups[j];

                if (!eventGroup[numEventGroup])
                {
                    printf("\nError: eventGroup initialization failed.\n");
                    exit(EXIT_FAILURE);
                }

                numEventGroup++;
            }
        }

        CUPTI_API_CALL(cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

        // Allocate Memory.
        for (i = 0; i < NUM_STREAMS; i++)
        {
            RUNTIME_API_CALL(cudaMalloc((void **)&devBufferA[i], bufferSize));

            pHostBuffer[i] = (float *)malloc(bufferSize);
            MEMORY_ALLOCATION_CALL(pHostBuffer[i]);
        }

        if (cpuToGpu)
        {
            TestCpuToGpu(eventGroup, devBufferA, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
            printf("Data transferred between CPU & Device %d: \n", (int)nvlinkRec->typeDev0);
        }
        else if (gpuToGpu)
        {
            RUNTIME_API_CALL(cudaSetDevice(1));

            for(i = 0; i < NUM_STREAMS; i++)
            {
                RUNTIME_API_CALL(cudaMalloc((void **)&devBufferB[i], bufferSize));
            }

            TestGpuToGpu(eventGroup, devBufferA, devBufferB, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
            printf("Data transferred between Device 0 & Device 1: \n");
        }

        // Collect Nvlink Metric values for the data transfer via Nvlink for all the eventGroups.
        for (i = 0; i < numEventGroup; i++)
        {
            ReadMetricValue(eventGroup[i], NUM_EVENTS, 0, metricId, timeDuration, metricValue);

            CUPTI_API_CALL(cuptiEventGroupDisable(eventGroup[i]));
            CUPTI_API_CALL(cuptiEventGroupDestroy(eventGroup[i]));

            for (i = 0; i < NUM_METRIC; i++)
            {
                if (PrintMetricValue(metricId[i], metricValue[i], pMetricName[i]) != 0)
                {
                    printf("\nError: PrintMetricValue() failed.\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    else
    {
        printf("Warning: No NvLink supported device found.\nWaiving.\n");
        exit(EXIT_WAIVED);
    }

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
