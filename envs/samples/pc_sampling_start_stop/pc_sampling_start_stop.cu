/*
 * Copyright 2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of pc sampling
 * with start stop APIs. This app will work on devices with compute
 * capability 7.0 and higher.
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <unordered_set>

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>

#define NUM_PC_COLLECT 100
#define ARRAY_SIZE 32000
#define THREADS_PER_BLOCK 256

typedef enum
{
    VECTOR_ADD  = 0,
    VECTOR_SUB  = 1,
    VECTOR_MUL  = 2,
} VectorOperation;

/// Kernels
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

__global__ void
VectorMultiply(
    const int *pA,
    const int *pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] * pB[i];
    }
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

static void
DoVectorOperation(
    const VectorOperation vectorOperation)
{
    int N = ARRAY_SIZE;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = 0;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    size_t size = N * sizeof(int);
    int i, result = 0;

    CUcontext cuCtx;

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

    cuCtxGetCurrent(&cuCtx);

    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (vectorOperation == VECTOR_ADD)
    {
        printf("Launching VectorAdd\n");
        VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else if (vectorOperation == VECTOR_SUB)
    {
        printf("Launching VectorSubtract\n");
        VectorSubtract <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else if (vectorOperation == VECTOR_MUL)
    {
        printf("Launching VectorMultiply\n");
        VectorMultiply <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else
    {
        fprintf(stderr, "Error: invalid operation\n");
        exit(EXIT_FAILURE);
    }

    RUNTIME_API_CALL(cudaMemcpyAsync(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Verify result
    for (i = 0; i < N; ++i)
    {
        if (vectorOperation == VECTOR_ADD)
        {
            result = pHostA[i] + pHostB[i];
        }
        else if (vectorOperation == VECTOR_SUB)
        {
            result = pHostA[i] - pHostB[i];
        }
        else if (vectorOperation == VECTOR_MUL)
        {
            result = pHostA[i] * pHostB[i];
        }
        else
        {
            fprintf(stderr, "Error: Invalid operation.\n");
            exit(EXIT_FAILURE);
        }

        if (pHostC[i] != result)
        {
            fprintf(stderr, "Error: Result verification failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
}

inline const char *
GetStallReason(
    const uint32_t& stallReasonCount,
    const uint32_t& pcSamplingStallReasonIndex,
    uint32_t *pStallReasonIndex,
    char **ppStallReasons)
{
    for (uint32_t i = 0; i < stallReasonCount; i++)
    {
        if (pStallReasonIndex[i] == pcSamplingStallReasonIndex)
        {
            return ppStallReasons[i];
        }
    }

    return "ERROR_STALL_REASON_INDEX_NOT_FOUND";
}

void
PrintPCSamplingData(
    CUpti_PCSamplingData *pPcSamplingData,
    const uint32_t& stallReasonCount,
    uint32_t *pStallReasonIndex,
    char **ppStallReasons)
{
    std::cout << "----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----" << std::endl;
    for (size_t i = 0; i < pPcSamplingData->totalNumPcs; i++)
    {
        std::cout << ", pcOffset : 0x"<< std::hex << pPcSamplingData->pPcData[i].pcOffset
                  << ", stallReasonCount: " << std::dec << pPcSamplingData->pPcData[i].stallReasonCount
                  << ", functionName: " << pPcSamplingData->pPcData[i].functionName;
        for (size_t j = 0; j < pPcSamplingData->pPcData[i].stallReasonCount; j++)
        {
            std::cout << ", stallReason: " << GetStallReason(stallReasonCount, pPcSamplingData->pPcData[i].stallReason[j].pcSamplingStallReasonIndex, pStallReasonIndex, ppStallReasons)
                      << ", samples: " << pPcSamplingData->pPcData[i].stallReason[j].samples;
        }
        std::cout << std::endl;
    }

    std::cout << "Number of PCs remaining to be collected: " << pPcSamplingData->remainingNumPcs << ", ";
    std::cout << "range id: " << pPcSamplingData->rangeId << ", ";
    std::cout << "total samples: " << pPcSamplingData->totalSamples << ", ";
    std::cout << "dropped samples: " << pPcSamplingData->droppedSamples << ", ";
    std::cout << "non user kernels total samples: " << pPcSamplingData->nonUsrKernelsTotalSamples << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;
}

bool
Run(
    const CUcontext& cuCtx,
    const size_t& stallReasonCount,
    uint32_t *pStallReasonIndex,
    char **ppStallReasons)
{
    CUpti_PCSamplingStartParams pcSamplingStartParams = {};
    pcSamplingStartParams.size = CUpti_PCSamplingStartParamsSize;
    pcSamplingStartParams.ctx = cuCtx;

    CUpti_PCSamplingStopParams pcSamplingStopParams = {};
    pcSamplingStopParams.size = CUpti_PCSamplingStopParamsSize;
    pcSamplingStopParams.ctx = cuCtx;

    // On-demand user buffer to hold collected PC Sampling data in PC-To-Counter format
    CUpti_PCSamplingData pcSamplingData;
    pcSamplingData.size = sizeof(CUpti_PCSamplingData);
    pcSamplingData.collectNumPcs = NUM_PC_COLLECT;
    pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)calloc(pcSamplingData.collectNumPcs, sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData);
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)calloc(stallReasonCount, sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData[i].stallReason);
    }

    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
    pcSamplingGetDataParams.ctx = cuCtx;
    pcSamplingGetDataParams.pcSamplingData = (void *)&pcSamplingData;

    // Kernel outside PC Sampling data collection range
    DoVectorOperation(VECTOR_MUL);

    // Start PC Sampling
    std::cout << "----- PC sampling start -----" << std::endl;
    CUPTI_API_CALL(cuptiPCSamplingStart(&pcSamplingStartParams));
    DoVectorOperation(VECTOR_ADD);
    DoVectorOperation(VECTOR_SUB);

    // Stop PC Sampling
    CUPTI_API_CALL(cuptiPCSamplingStop(&pcSamplingStopParams));
    std::cout << "----- PC sampling stop -----" << std::endl;

    // Collect PC Sampling data
    CUPTI_API_CALL(cuptiPCSamplingGetData(&pcSamplingGetDataParams));
    PrintPCSamplingData(&pcSamplingData, stallReasonCount, pStallReasonIndex, ppStallReasons);

    // Kernel outside PC Sampling data collection range
    DoVectorOperation(VECTOR_MUL);

    // Start PC Sampling
    std::cout << "----- PC sampling start -----" << std::endl;
    CUPTI_API_CALL(cuptiPCSamplingStart(&pcSamplingStartParams));
    DoVectorOperation(VECTOR_MUL);
    DoVectorOperation(VECTOR_ADD);
    DoVectorOperation(VECTOR_SUB);

    // Stop PC Sampling
    CUPTI_API_CALL(cuptiPCSamplingStop(&pcSamplingStopParams));
    std::cout << "----- PC sampling stop -----" << std::endl;

    // Collect PC Sampling data
    CUPTI_API_CALL(cuptiPCSamplingGetData(&pcSamplingGetDataParams));
    PrintPCSamplingData(&pcSamplingData, stallReasonCount, pStallReasonIndex, ppStallReasons);

    // Kernel outside PC Sampling data collection range
    DoVectorOperation(VECTOR_MUL);

    // Free memory
    std::unordered_set<char*> functions;
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        if (pcSamplingData.pPcData[i].stallReason)
        {
            free(pcSamplingData.pPcData[i].stallReason);
        }

        if (pcSamplingData.pPcData[i].functionName)
        {
            functions.insert(pcSamplingData.pPcData[i].functionName);
        }
    }

    for(auto it = functions.begin(); it != functions.end(); ++it)
    {
        free(*it);
    }
    functions.clear();

    if (pcSamplingData.pPcData)
    {
        free(pcSamplingData.pPcData);
    }
    return true;
}

void
DoPCSampling(
    const CUcontext& cuCtx)
{
    // Enable PC Sampling
    CUpti_PCSamplingEnableParams pcSamplingEnableParams = {};
    pcSamplingEnableParams.size = CUpti_PCSamplingEnableParamsSize;
    pcSamplingEnableParams.ctx = cuCtx;
    CUPTI_API_CALL(cuptiPCSamplingEnable(&pcSamplingEnableParams));

    // Get number of supported counters
    size_t numStallReasons = 0;
    CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {};
    numStallReasonsParams.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    numStallReasonsParams.ctx = cuCtx;
    numStallReasonsParams.numStallReasons = &numStallReasons;
    CUPTI_API_CALL(cuptiPCSamplingGetNumStallReasons(&numStallReasonsParams));

    char **ppStallReasons = (char **)calloc(numStallReasons, sizeof(char*));
    MEMORY_ALLOCATION_CALL(ppStallReasons);
    for (size_t i = 0; i < numStallReasons; i++)
    {
        ppStallReasons[i] = (char *)calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char));
        MEMORY_ALLOCATION_CALL(ppStallReasons[i]);
    }
    uint32_t *pStallReasonIndex = (uint32_t *)calloc(numStallReasons, sizeof(uint32_t));
    MEMORY_ALLOCATION_CALL(pStallReasonIndex);

    CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {};
    stallReasonsParams.size = CUpti_PCSamplingGetStallReasonsParamsSize;
    stallReasonsParams.ctx = cuCtx;
    stallReasonsParams.numStallReasons = numStallReasons;
    stallReasonsParams.stallReasonIndex = pStallReasonIndex;
    stallReasonsParams.stallReasons = ppStallReasons;
    CUPTI_API_CALL(cuptiPCSamplingGetStallReasons(&stallReasonsParams));

    // Buffer to hold collected PC Sampling data in PC-To-Counter format
    CUpti_PCSamplingData pcSamplingData;
    pcSamplingData.size = sizeof(CUpti_PCSamplingData);
    pcSamplingData.collectNumPcs = NUM_PC_COLLECT;
    pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)calloc(pcSamplingData.collectNumPcs, sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData);
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)calloc(numStallReasons, sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(pcSamplingData.pPcData[i].stallReason);
    }

    // PC Sampling configuration attributes
    CUpti_PCSamplingConfigurationInfo enableStartStop = {};
    enableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    enableStartStop.attributeData.enableStartStopControlData.enableStartStopControl = true;

    CUpti_PCSamplingConfigurationInfo samplingDataBuffer = {};
    samplingDataBuffer.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    samplingDataBuffer.attributeData.samplingDataBufferData.samplingDataBuffer = (void *)&pcSamplingData;

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
    pcSamplingConfigurationInfo.push_back(enableStartStop);
    pcSamplingConfigurationInfo.push_back(samplingDataBuffer);

    CUpti_PCSamplingConfigurationInfoParams pcSamplingConfigurationInfoParams = {};
    pcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    pcSamplingConfigurationInfoParams.ctx = cuCtx;
    pcSamplingConfigurationInfoParams.numAttributes = pcSamplingConfigurationInfo.size();
    pcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingConfigurationInfo.data();

    CUPTI_API_CALL(cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    if(!Run(cuCtx, numStallReasons, pStallReasonIndex, ppStallReasons))
    {
        std::cout << "Failed to run sample" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Disable PC Sampling
    CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
    pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
    pcSamplingDisableParams.ctx = cuCtx;
    CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

    // Free memory
    for (size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        if (pcSamplingData.pPcData[i].stallReason)
        {
            free(pcSamplingData.pPcData[i].stallReason);
        }
    }
    if (pcSamplingData.pPcData)
    {
        free(pcSamplingData.pPcData);
    }

    for (size_t i = 0; i < numStallReasons; i++)
    {
        if (ppStallReasons[i])
        {
            free(ppStallReasons[i]);
        }
    }
    if (ppStallReasons)
    {
        free(ppStallReasons);
    }
}

int
main(
    int argc,
    char *argv[])
{
    CUcontext cuCtx;
    cudaDeviceProp prop;
    int deviceNum = 0;

    RUNTIME_API_CALL(cudaGetDevice(&deviceNum));

    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    printf("Device Name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);

    // Initialize profiler API and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Sample is waived on this device, Unable to profile on device " << deviceNum << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    cuCtxGetCurrent(&cuCtx);
    DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, deviceNum));

    DoPCSampling(cuCtx);

    DRIVER_API_CALL(cuCtxDestroy(cuCtx));

    exit(EXIT_SUCCESS);
}
