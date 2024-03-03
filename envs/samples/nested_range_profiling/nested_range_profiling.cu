// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// The sample provides workflow for adding nested ranges for profiling with CUPTI profiling APIs.
// The psuedo code for the sample
// cuptiProfilerPushRange(rangeA)           // push rangeA -> nesting level 1
//    launch kernel A
//    cuptiProfilerPushRange(rangeB)        // push rangeB -> nesting level 2
//        launch kernel B
//    cuptiProfilerPopRange()               // pop rangeB
// cuptiProfilerPopRange()                  // pop rangeA
//
// Notes:
// 1) Even though nested ranges are subset of user range they still count as individual range,
//    so the number of ranges need to be sum of user ranges and nested ranges and as the ranges number increases the profiling time also increases.
// 2) The number of passes required for collecting counter data will vary with number of nesting level used.
//    e.g. "sm__mio_inst_issued.sum" metric in GA100 needs 2 passes for collecting the counter data with no nesting (nestingLevel = 1).
//    if we add a nested range (nestingLevel = 2) then the number of passes required will be 2 times. (4 passes)
//    (You can refer to cupti_metric_properties sample for getting the metrics properties like number of passes required and
//    which type of metric it is (HW/SW) for a particular chip)
//

// Standard STL headers
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_profiler_target.h>

// NVPW headers
#include <nvperf_host.h>

// Make use of example code wrappers for NVPW calls
#include <Eval.h>
#include <Utils.h>
#include <Metric.h>
#include <FileOp.h>

// Global emums and variables
static int s_NumRanges = 2;
static int s_NumNestingLevels = 2;

enum class eVectorOperationType
{
    VEC_ADD,
    VEC_SUB
};

// Macros
#define DEFAULT_METRIC_NAME "sm__ctas_launched.sum"

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

static void
ProcessVector(
    int numOfElements,
    eVectorOperationType operationType)
{
    size_t size = numOfElements * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    int i, res;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors.
    InitializeVector(pHostA, numOfElements);
    InitializeVector(pHostB, numOfElements);
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
    blocksPerGrid = (numOfElements + threadsPerBlock - 1) / threadsPerBlock;

    if (operationType == eVectorOperationType::VEC_ADD)
    {
        printf("Launching VecAdd kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);
        VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, numOfElements);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    if (operationType == eVectorOperationType::VEC_SUB)
    {
        printf("Launching VecSub kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);
        VectorSubtract <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, numOfElements);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < numOfElements; ++i)
    {
        if (operationType == eVectorOperationType::VEC_ADD)
            res = pHostA[i] + pHostB[i];

        if (operationType == eVectorOperationType::VEC_SUB)
            res = pHostA[i] - pHostB[i];

        if (pHostC[i] != res)
        {
            fprintf(stderr, "error: result verification failed\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
}

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = s_NumRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = s_NumRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

bool RunTest(
    std::vector<uint8_t>& configImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImage,
    CUpti_ProfilerReplayMode profilerReplayMode,
    CUpti_ProfilerRange profilerRange)
{
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.ctx = cuContext;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = s_NumRanges;
    beginSessionParams.maxLaunchesPerPass = s_NumRanges;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = s_NumNestingLevels;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    // User takes the resposiblity of replaying the kernel launches.
    CUpti_Profiler_BeginPass_Params beginPassParams = { CUpti_Profiler_BeginPass_Params_STRUCT_SIZE };
    CUpti_Profiler_EndPass_Params endPassParams = { CUpti_Profiler_EndPass_Params_STRUCT_SIZE };
    do
    {
        CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));

        {
            CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
            CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

            CUpti_Profiler_PushRange_Params pushRangeParams = { CUpti_Profiler_PushRange_Params_STRUCT_SIZE };
            pushRangeParams.pRangeName = "userRangeA";
            printf("\nStart of userRangeA\n");
            CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
            {
                ProcessVector(50000, eVectorOperationType::VEC_ADD);
                // Nested range start.
                pushRangeParams.pRangeName = "userRangeB";
                printf("Start of userRangeB\n");
                CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
                {
                    ProcessVector(10000, eVectorOperationType::VEC_SUB);
                }
                CUpti_Profiler_PopRange_Params popRangeParams = { CUpti_Profiler_PopRange_Params_STRUCT_SIZE };
                printf("End of userRangeB\n");
                CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
                // Nested range End.
            }

            CUpti_Profiler_PopRange_Params popRangeParams = { CUpti_Profiler_PopRange_Params_STRUCT_SIZE };
            printf("End of userRangeA\n");
            CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));

            CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
        }

        CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));

    } while (!endPassParams.allPassesSubmitted);

    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    return true;
}

int main(
    int argc,
    char *argv[])
{
    CUdevice cuDevice;
    std::vector<std::string> metricNames;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
    std::vector<uint8_t> counterAvailabilityImage;
    std::string CounterDataFileName("SimpleCupti.counterdata");
    std::string CounterDataSBFileName("SimpleCupti.counterdataSB");
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_UserRange;
    char* metricName;
    int deviceCount, deviceNum;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    printf("Usage: %s [device_num] [metric_names comma separated]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
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

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor,computeCapabilityMinor);

    // Initialize profiler API and test device compatibility.
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;

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

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    // Get the names of the metrics to collect.
    if (argc > 2)
    {
        metricName = strtok(argv[2], ",");
        while(metricName != NULL)
        {
            metricNames.push_back(metricName);
            metricName = strtok(NULL, ",");
        }
    }
    else {
        metricNames.push_back(DEFAULT_METRIC_NAME);
    }

    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

    // Get chip name for the cuda  device.
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    std::string chipName(getChipNameParams.pChipName);

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = cuContext;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    counterAvailabilityImage.clear();
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Generate configuration for metrics, this can also be done offline.
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    RETURN_IF_NVPW_ERROR(0, NVPW_InitializeHost(&initializeHostParams));

    if (metricNames.size())
    {
        if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage, counterAvailabilityImage.data()))
        {
            std::cerr << "Failed to create the ConfigImage." << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
        {
            std::cerr << "Failed to create the CounterDataPrefixImage." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "No metrics provided to profile." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix))
    {
        std::cerr << "  Failed to create the CounterDataImage." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!RunTest(configImage, counterDataScratchBuffer, counterDataImage, profilerReplayMode, profilerRange))
    {
        std::cerr << "Failed to run the sample." << std::endl;
        exit(EXIT_FAILURE);
    }

    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
    DRIVER_API_CALL(cuCtxDestroy(cuContext));

    // Dump counterDataImage in file.
    WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);

    // Evaluation of metrics collected in counterDataImage, this can also be done offline.
    NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);
    exit(EXIT_SUCCESS);
}
