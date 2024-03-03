//
// Copyright 2020-2022 NVIDIA Corporation. All rights reserved
//

// System headers
#include <string>
#include <stdio.h>
#include <stdlib.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_profiler_target.h>

// Perfworks headers
#include <nvperf_host.h>

// Make use of example code wrappers for NVPW calls
#include <Eval.h>
#include <Metric.h>
#include <FileOp.h>

// Global variables
static int s_NumRanges = 2;

// Macros
#define METRIC_NAME "smsp__warps_launched.avg"

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
    int *pHostD,
    int *pDeviceA,
    int *pDeviceB,
    int *pDeviceC,
    int *pDeviceD)
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
    if (pHostD)
    {
        free(pHostD);
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
    if (pDeviceD)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceD));
    }
}

static void
DoVectorAddSubtract()
{
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *pHostA, *pHostB, *pHostC, *pHostD;
    int *pDeviceA, *pDeviceB, *pDeviceC, *pDeviceD;
    int i, sum, diff;

    // Allocate input vectors pHostA and pHostB in host memory
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    pHostD = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostD);

    // Initialize input vectors
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);
    memset(pHostD, 0, size);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceD, size));

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());
    VectorSubtract <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceD, N);
    RUNTIME_API_CALL(cudaGetLastError());

    // Copy result from device memory to host memory
    // pHostC contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));
    RUNTIME_API_CALL(cudaMemcpy(pHostD, pDeviceD, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        sum  = pHostA[i] + pHostB[i];
        diff = pHostA[i] - pHostB[i];
        if (pHostC[i] != sum || pHostD[i] != diff)
        {
            fprintf(stderr, "Error: Result verification failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pHostD, pDeviceA, pDeviceB, pDeviceC, pDeviceD);
}

bool
CreateCounterDataImage(
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

bool
RunTest(
    std::vector<uint8_t>& configImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImage,
    CUpti_ProfilerReplayMode profilerReplayMode,
    CUpti_ProfilerRange profilerRange)
{
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_BeginSession_Params beginSessionParams         = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    CUpti_Profiler_SetConfig_Params setConfigParams               = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams   = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = s_NumRanges;
    beginSessionParams.maxLaunchesPerPass = s_NumRanges;

    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    // Profile in KernelReplayMode.
    if (profilerReplayMode == CUPTI_KernelReplay)
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
        DoVectorAddSubtract();
        CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    }
    // Profile in UserReplayMode.
    else if (profilerReplayMode == CUPTI_UserReplay)
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

        // User takes the resposiblity of replaying the kernel launches.
        CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
        CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};

        do
        {
            CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
            {
                CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
                DoVectorAddSubtract();
                CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            }
            CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
        }
        while (!endPassParams.allPassesSubmitted);

        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    return true;
}

int
main(
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
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    int deviceCount, deviceNum;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    char* pMetricName;

    printf("Usage: %s [device_num] [metric_names comma separated]\n", argv[0]);

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

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    // Initialize profiler API support and test device compatibility
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
        pMetricName = strtok(argv[2], ",");
        while (pMetricName != NULL)
        {
            metricNames.push_back(pMetricName);
            pMetricName = strtok(NULL, ",");
        }
    }
    else
    {
        metricNames.push_back(METRIC_NAME);
    }

    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

    // Get chip name for the CUDA device.
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
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (metricNames.size())
    {
        if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage, counterAvailabilityImage.data()))
        {
            std::cout << "Error: Failed to create the ConfigImage." << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
        {
            std::cout << "Error: Failed to create the CounterDataPrefixImage." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout << "Error: No metrics provided to profile." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix))
    {
        std::cout << "Error: Failed to create the CounterDataImage." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!RunTest(configImage, counterDataScratchBuffer, counterDataImage, profilerReplayMode, profilerRange))
    {
        std::cout << "Error: Failed to run the sample." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    DRIVER_API_CALL(cuCtxDestroy(cuContext));

    // Dump counterDataImage in file.
    WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);

    // Evaluation of metrics collected in counterDataImage.
    // This can also be done offline.
    NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);

    exit(EXIT_SUCCESS);
}
