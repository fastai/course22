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
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_profiler_target.h>

// Perfworks headers
#include <nvperf_host.h>

// Make use of example code wrappers for NVPW calls.
#include <Eval.h>
#include <Metric.h>
#include <FileOp.h>

// Structures
typedef struct ProfilingData_t
{
    int numRanges = 2;
    bool bProfiling = false;
    std::string chipName;
    std::vector<std::string> metricNames;
    std::string counterDataFileName = "SimpleCupti.counterdata";
    std::string counterDataSBFileName = "SimpleCupti.counterdataSB";
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    bool allPassesSubmitted = true;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
} ProfilingData;

// Macros
#define METRIC_NAME "sm__ctas_launched.sum"

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
void
EnableProfiling(
    ProfilingData *pProfilingData)
{
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay)
    {
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    }
    else if (pProfilingData->profilerReplayMode == CUPTI_UserReplay)
    {
        CUpti_Profiler_BeginPass_Params beginPassParams = { CUpti_Profiler_BeginPass_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    }
}

void
DisableProfiling(
    ProfilingData *pProfilingData)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    if (pProfilingData->profilerReplayMode == CUPTI_UserReplay)
    {
        CUpti_Profiler_EndPass_Params endPassParams = { CUpti_Profiler_EndPass_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
        pProfilingData->allPassesSubmitted = (endPassParams.allPassesSubmitted == 1) ? true : false;
    }
    else if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay)
    {
        pProfilingData->allPassesSubmitted = true;
    }

    if (pProfilingData->allPassesSubmitted)
    {
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = { CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }
}

void
BeginSession(
    ProfilingData *pProfilingData)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = pProfilingData->counterDataImage.size();
    beginSessionParams.pCounterDataImage = &pProfilingData->counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = pProfilingData->counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &pProfilingData->counterDataScratchBuffer[0];
    beginSessionParams.range = pProfilingData->profilerRange;
    beginSessionParams.replayMode = pProfilingData->profilerReplayMode;
    beginSessionParams.maxRangesPerPass = pProfilingData->numRanges;
    beginSessionParams.maxLaunchesPerPass = pProfilingData->numRanges;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
}

void
SetConfig(
    ProfilingData *pProfilingData)
{
    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = &pProfilingData->configImage[0];
    setConfigParams.configSize = pProfilingData->configImage.size();
    setConfigParams.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
}

void
CreateCounterDataImage(
    int numRanges,
    std::vector<uint8_t>& counterDataImagePrefix,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImage)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));
}

void
SetupProfiling(
    ProfilingData *pProfilingData)
{
    // Generate configuration for metrics, this can also be done offline.
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (pProfilingData->metricNames.size())
    {
        if (!NV::Metric::Config::GetConfigImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->configImage))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!NV::Metric::Config::GetCounterDataPrefixImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(EXIT_FAILURE);
    }

    CreateCounterDataImage(pProfilingData->numRanges, pProfilingData->counterDataImagePrefix,
                           pProfilingData->counterDataScratchBuffer, pProfilingData->counterDataImage);

    BeginSession(pProfilingData);
    SetConfig(pProfilingData);
}

void
StopProfiling(
    ProfilingData *pProfilingData)
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};

    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    // Dump counterDataImage and counterDataScratchBuffer in file.
    WriteBinaryFile(pProfilingData->counterDataFileName.c_str(), pProfilingData->counterDataImage);
    WriteBinaryFile(pProfilingData->counterDataSBFileName.c_str(), pProfilingData->counterDataScratchBuffer);
}

void
ProfilingCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData)
{
    ProfilingData *pProfilingData = (ProfilingData *)(pUserData);
    const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {
            switch (callbackId)
            {
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
                {
                    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
                    {
                        EnableProfiling(pProfilingData);
                    }
                    else
                    {
                        DisableProfiling(pProfilingData);
                    }
                }
                break;
                default:
                    break;
            }
            break;
        }
        case CUPTI_CB_DOMAIN_RESOURCE:
        {
            switch (callbackId)
            {
                case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
                {
                    SetupProfiling(pProfilingData);
                    pProfilingData->bProfiling = true;
                }
                break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

}

void
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

void
DoVectorAddition()
{
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int* pHostA, * pHostB, * pHostC;
    int* pDeviceA, * pDeviceB, * pDeviceC;
    int i, sum;

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
            fprintf(stderr, "Error: result verification failed\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
}

int
main(
    int argc,
    char *argv[])
{
    CUdevice cuDevice = 0;
    int deviceCount, deviceNum = 0;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    // Initialize profiler API support and test device compatibility.
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
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

    ProfilingData *pProfilingData = new ProfilingData();
    for (int i = 1; i < argc; ++i)
    {
        char* arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
        {
            printf("Usage: %s -d [device_num] -m [metric_names comma separated] -n [num of ranges] -r [kernel or user] -o [counterdata filename]\n", argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (strcmp(arg, "--device") == 0 || strcmp(arg, "-d") == 0)
        {
            deviceNum = atoi(argv[i + 1]);
            printf("CUDA Device Number: %d\n", deviceNum);
            i++;
        }
        else if (strcmp(arg, "--metrics") == 0 || strcmp(arg, "-m") == 0)
        {
            char* metricName = strtok(argv[i + 1], ",");
            while (metricName != NULL)
            {
                pProfilingData->metricNames.push_back(metricName);
                metricName = strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--numRanges") == 0 || strcmp(arg, "-n") == 0)
        {
            int numRanges = atoi(argv[i + 1]);
            pProfilingData->numRanges = numRanges;
            i++;
        }
        else if (strcmp(arg, "--replayMode") == 0 || strcmp(arg, "-r") == 0)
        {
            std::string replayMode(argv[i + 1]);
            if (replayMode == "kernel")
            {
                pProfilingData->profilerReplayMode = CUPTI_KernelReplay;
            }
            else if (replayMode == "user")
            {
                pProfilingData->profilerReplayMode = CUPTI_UserReplay;
            }
            else
            {
                printf("Invalid --replayMode argument supported replayMode type 'kernel' or 'user'\n");
                exit(EXIT_FAILURE);
            }
            i++;
        }
        else if (strcmp(arg, "--outputCounterData") == 0 || strcmp(arg, "-o") == 0)
        {
            std::string outputCounterData(argv[i + 1]);
            pProfilingData->counterDataFileName = outputCounterData;
            pProfilingData->counterDataSBFileName = outputCounterData + "SB";
            i++;
        }
        else
        {
            printf("Error!! Invalid Arguments\n");
            printf("Usage: %s -d [device_num] -m [metric_names comma separated] -n [num of ranges] -r [kernel or user] -o [counterdata filename]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (pProfilingData->metricNames.empty())
    {
        pProfilingData->metricNames.push_back(METRIC_NAME);
    }

    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    pProfilingData->chipName = getChipNameParams.pChipName;

    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)ProfilingCallbackHandler, pProfilingData));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));

    do
    {
        DoVectorAddition();
    }
    while (!pProfilingData->allPassesSubmitted);

    if (pProfilingData->bProfiling)
    {
        StopProfiling(pProfilingData);
        pProfilingData->bProfiling = false;

        // Evaluation of metrics collected in counterDataImage, this can also be done offline.
        NV::Metric::Eval::PrintMetricValues(pProfilingData->chipName, pProfilingData->counterDataImage, pProfilingData->metricNames);
    }

    delete pProfilingData;

    exit(EXIT_SUCCESS);
}
