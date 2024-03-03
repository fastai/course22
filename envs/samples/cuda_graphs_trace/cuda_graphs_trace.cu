/*
 * Copyright 2020-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print the trace of CUDA graphs and correlate
 * the graph node launch to the node creation API using CUPTI callbacks.
 */

 // System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Macros
#define COMPUTE_N 50000

// Data structures
typedef struct ApiData_st
{
    const char *pFunctionName;
    uint32_t correlationId;
} ApiData;

typedef std::map<uint64_t, ApiData> NodeIdApiDataMap;
NodeIdApiDataMap nodeIdCorrelationMap;

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

__global__ void
VectorSubtract(
    const int *pA,
    const int *pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] - pB[i];
    }
}

// Functions
static void
DoPass(
    cudaStream_t stream)
{
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;

    size_t size = COMPUTE_N * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;

    cudaKernelNodeParams kernelParams;
    cudaMemcpy3DParms memcpyParams = {0};
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t nodes[5];

    // Allocate input vectors pHostA and pHostB in host memory.
    // Don't bother to initialize.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    RUNTIME_API_CALL(cudaGraphCreate(&graph, 0));

    // Initialize memcpy params.
    memcpyParams.kind = cudaMemcpyHostToDevice;
    memcpyParams.srcPtr.ptr = pHostA;
    memcpyParams.dstPtr.ptr = pDeviceA;
    memcpyParams.extent.width = size;
    memcpyParams.extent.height = 1;
    memcpyParams.extent.depth = 1;
    RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[0], graph, NULL, 0, &memcpyParams));

    memcpyParams.srcPtr.ptr = pHostB;
    memcpyParams.dstPtr.ptr = pDeviceB;
    RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[1], graph, NULL, 0, &memcpyParams));

    // Initialize kernel params.
    int num = COMPUTE_N;
    void* kernelArgs[] = {(void *)&pDeviceA, (void *)&pDeviceB, (void *)&pDeviceC, (void *)&num};
    blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
    kernelParams.func = (void *)VectorAdd;
    kernelParams.gridDim = dim3(blocksPerGrid, 1, 1);
    kernelParams.blockDim = dim3(threadsPerBlock, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = (void **)kernelArgs;
    kernelParams.extra = NULL;

    RUNTIME_API_CALL(cudaGraphAddKernelNode(&nodes[2], graph, &nodes[0], 2, &kernelParams));

    kernelParams.func = (void *)VectorSubtract;
    RUNTIME_API_CALL(cudaGraphAddKernelNode(&nodes[3], graph, &nodes[2], 1, &kernelParams));

    memcpyParams.kind = cudaMemcpyDeviceToHost;
    memcpyParams.srcPtr.ptr = pDeviceC;
    memcpyParams.dstPtr.ptr = pHostC;
    memcpyParams.extent.width = size;
    memcpyParams.extent.height = 1;
    memcpyParams.extent.depth = 1;
    RUNTIME_API_CALL(cudaGraphAddMemcpyNode(&nodes[4], graph, &nodes[3], 1, &memcpyParams));

    RUNTIME_API_CALL(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    RUNTIME_API_CALL(cudaGraphLaunch(graphExec, stream));
    RUNTIME_API_CALL(cudaStreamSynchronize(stream));

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

void
GraphTraceRecords(
    CUpti_Activity *pRecord)
{
    switch (pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy5 *pMemcpyRecord = (CUpti_ActivityMemcpy5 *) pRecord;

            // Retrieve the information of the API used to create the node.
            NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(pMemcpyRecord->graphNodeId);
            if (it != nodeIdCorrelationMap.end())
            {
                printf("Graph node was created using API %s with correlationId %u\n", it->second.pFunctionName, it->second.correlationId);
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel9 *pKernelRecord = (CUpti_ActivityKernel9 *) pRecord;

            // Retrieve the information of the API used to create the node.
            NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(pKernelRecord->graphNodeId);
            if (it != nodeIdCorrelationMap.end())
            {
                printf("Graph node was created using API %s with correlationId %u\n", it->second.pFunctionName, it->second.correlationId);
            }

            break;
        }
        default:
            break;
    }
}

void CUPTIAPI
GraphsCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const CUpti_CallbackData *pCallbackInfo)
{
    static const char *s_pFunctionName;
    static uint32_t s_correlationId;

    // Check last error.
    CUPTI_API_CALL(cuptiGetLastError());

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_RESOURCE:
        {
            CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)pCallbackInfo;
            switch (callbackId)
            {
                case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED:
                {
                    // Do not store info for the nodes that are created during graph instantiate.
                    if (!strncmp(s_pFunctionName, "cudaGraphInstantiate", strlen("cudaGraphInstantiate")))
                    {
                        break;
                    }
                    CUpti_GraphData *callbackData = (CUpti_GraphData *) pResourceData->resourceDescriptor;
                    uint64_t nodeId;

                    // Query the graph node ID and store the API correlation id and function name.
                    CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->node, &nodeId));
                    ApiData apiData;
                    apiData.correlationId = s_correlationId;
                    apiData.pFunctionName = s_pFunctionName;
                    nodeIdCorrelationMap[nodeId] = apiData;
                    break;
                }
                case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED:
                {
                    CUpti_GraphData *callbackData = (CUpti_GraphData *) pResourceData->resourceDescriptor;
                    uint64_t nodeId, originalNodeId;

                    // Overwrite the map entry with node ID of the cloned graph node.
                    CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->originalNode, &originalNodeId));
                    NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(originalNodeId);
                    if (it != nodeIdCorrelationMap.end())
                    {
                        CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->node, &nodeId));
                        ApiData apiData = it->second;
                        nodeIdCorrelationMap.erase(it);
                        nodeIdCorrelationMap[nodeId] = apiData;
                    }
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case CUPTI_CB_DOMAIN_RUNTIME_API:
        {
            if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
            {
                s_correlationId = pCallbackInfo->correlationId;
                s_pFunctionName = pCallbackInfo->functionName;
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
    pUserData->pPostProcessActivityRecords = GraphTraceRecords;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, (void *)GraphsCallbackHandler, stdout);

    // Enable activity record kinds.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // Enable callbacks for CUDA graph.
    CUPTI_API_CALL(cuptiEnableCallback(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED));
    CUPTI_API_CALL(cuptiEnableCallback(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));
    CUPTI_API_CALL(cuptiEnableDomain(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API));
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

    // DoPass with user stream.
    cudaStream_t stream;
    RUNTIME_API_CALL(cudaStreamCreate(&stream));
    DoPass(stream);

    RUNTIME_API_CALL(cudaDeviceSynchronize());
    RUNTIME_API_CALL(cudaDeviceReset());

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
