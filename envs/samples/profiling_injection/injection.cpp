// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// This sample demostrates using the profiler API in injection mode.
// Build this file as a shared object, and set environment variable
// CUDA_INJECTION64_PATH to the full path to the .so.
//
// CUDA will load the object during initialization and will run
// the function called 'InitializeInjection'.
//
// After the initialization routine  returns, the application resumes running,
// with the registered callbacks triggering as expected.  These callbacks
// are used to start a Profiler API session using Kernel Replay and
// Auto Range modes.
//
// A configurable number of kernel launches (default 10) are run
// under one session.  Before the 11th kernel launch, the callback
// ends the session, prints metrics, and starts a new session.
//
// An atexit callback is also used to ensure that any partial sessions
// are handled when the target application exits.
//
// This code supports multiple contexts and multithreading through
// locking shared data structures.

// System headers
#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <unordered_map>
using ::std::unordered_map;

#include <unordered_set>
using ::std::unordered_set;

#include <vector>
using ::std::vector;

#include <stdlib.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUPTI headers
#include <cupti_callbacks.h>
#include <cupti_profiler_target.h>
#include <cupti_driver_cbid.h>
#include <cupti_target.h>
#include <cupti_activity.h>
#include "helper_cupti.h"

// NVPW headers
#include <nvperf_host.h>

#include <Eval.h>
using ::NV::Metric::Eval::PrintMetricValues;

#include <Metric.h>
using ::NV::Metric::Config::GetConfigImage;
using ::NV::Metric::Config::GetCounterDataPrefixImage;

#include <Utils.h>
using ::NV::Metric::Utils::GetNVPWResultString;

// Macros
// Export InitializeInjection symbol.
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

// Profiler API configuration data, per-context.
struct CtxProfilerData
{
    CUcontext       ctx;
    int             deviceId;
    cudaDeviceProp  deviceProp;
    vector<uint8_t> counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    vector<uint8_t> counterDataImage;
    vector<uint8_t> counterDataPrefixImage;
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
    // Count of sessions.
    int             iterations;

    // Initialize fields, with env var overrides.
    CtxProfilerData() : curRanges(), maxRangeNameLength(64), iterations()
    {
        char *pEnvVar = getenv("INJECTION_KERNEL_COUNT");
        if (pEnvVar != NULL)
        {
            int value = atoi(pEnvVar);
            if (value < 1)
            {
                cerr << "Read " << value << " kernels from INJECTION_KERNEL_COUNT, but must be >= 1; defaulting to 10." << endl;
                value = 10;
            }
            maxNumRanges = value;
        }
        else
        {
            maxNumRanges = 10;
        }
    };
};

// Track per-context profiler API data in a shared map.
mutex ctxDataMutex;
unordered_map<CUcontext, CtxProfilerData> contextData;

// List of metrics to collect.
vector<string> metricNames;

// Initialize state.
void
InitializeState()
{
    static int profilerInitialized = 0;

    if (profilerInitialized == 0)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // NVPW required initialization
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        profilerInitialized = 1;
    }
}

// Initialize profiler for a context.
void
InitializeContextData(
    CtxProfilerData &contextData)
{
    InitializeState();

    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data.
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Allocate sized counterAvailabilityImage.
    contextData.counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);

    // Initialize counterAvailabilityImage.
    getCounterAvailabilityParams.pCounterAvailabilityImage = contextData.counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Fill in configImage - can be run on host or target.
    if (!GetConfigImage(contextData.deviceProp.name, metricNames, contextData.configImage, contextData.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create configImage for context " << contextData.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Fill in counterDataPrefixImage - can be run on host or target.
    if (!GetCounterDataPrefixImage(contextData.deviceProp.name, metricNames, contextData.counterDataPrefixImage, contextData.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create counterDataPrefixImage for context " << contextData.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Record counterDataPrefixImage info and other options for sizing the counterDataImage.
    contextData.counterDataImageOptions.pCounterDataPrefix = contextData.counterDataPrefixImage.data();
    contextData.counterDataImageOptions.counterDataPrefixSize = contextData.counterDataPrefixImage.size();
    contextData.counterDataImageOptions.maxNumRanges = contextData.maxNumRanges;
    contextData.counterDataImageOptions.maxNumRangeTreeNodes = contextData.maxNumRanges;
    contextData.counterDataImageOptions.maxRangeNameLength = contextData.maxRangeNameLength;

    // Calculate size of counterDataImage based on counterDataPrefixImage and options.
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &(contextData.counterDataImageOptions);
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
    // Create counterDataImage.
    contextData.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    // Initialize counterDataImage inside StartSession.
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(contextData.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = contextData.counterDataImage.size();
    initializeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage.
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = contextData.counterDataImage.size();
    scratchBufferSizeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    // Create counterDataScratchBuffer
    contextData.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    // Initialize counterDataScratchBuffer.
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = contextData.counterDataImage.size();
    initScratchBufferParams.pCounterDataImage = contextData.counterDataImage.data();
    initScratchBufferParams.counterDataScratchBufferSize = contextData.counterDataScratchBufferImage.size();;
    initScratchBufferParams.pCounterDataScratchBuffer = contextData.counterDataScratchBufferImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

}

// Start a session.
void
StartSession(
    CtxProfilerData &contextData)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.counterDataImageSize = contextData.counterDataImage.size();
    beginSessionParams.pCounterDataImage = contextData.counterDataImage.data();
    beginSessionParams.counterDataScratchBufferSize = contextData.counterDataScratchBufferImage.size();
    beginSessionParams.pCounterDataScratchBuffer = contextData.counterDataScratchBufferImage.data();
    beginSessionParams.ctx = contextData.ctx;
    beginSessionParams.maxLaunchesPerPass = contextData.maxNumRanges;
    beginSessionParams.maxRangesPerPass = contextData.maxNumRanges;
    beginSessionParams.pPriv = NULL;
    beginSessionParams.range = CUPTI_AutoRange;
    beginSessionParams.replayMode = CUPTI_KernelReplay;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = contextData.configImage.data();
    setConfigParams.configSize = contextData.configImage.size();
    // Only set for Application Replay mode
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    enableProfilingParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

    contextData.iterations++;
}

// Print session data
static void
PrintData(
    CtxProfilerData &contextData)
{
    cout << endl << "Context " << contextData.ctx << ", device " << contextData.deviceId << " (" << contextData.deviceProp.name << ") session " << contextData.iterations << ":" << endl;
    PrintMetricValues(contextData.deviceProp.name, contextData.counterDataImage, metricNames, contextData.counterAvailabilityImage.data());
}

// End a session during execution
void
EndSession(
    CtxProfilerData &contextData)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    disableProfilingParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    unsetConfigParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    endSessionParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    PrintData(contextData);

    // Clear counterDataImage (otherwise it maintains previous records when it is reused)
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(contextData.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = contextData.counterDataImage.size();
    initializeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));
}

// Clean up at end of execution
static void
EndExecution()
{
    CUPTI_API_CALL(cuptiGetLastError());
    ctxDataMutex.lock();

    for (auto itr = contextData.begin(); itr != contextData.end(); ++itr)
    {
        CtxProfilerData &data = itr->second;

        if (data.curRanges > 0)
        {
            PrintData(data);
            data.curRanges = 0;
        }
    }

    ctxDataMutex.unlock();
}

// Callback handler
void
ProfilerCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void const *pCallbackData)
{
    static int initialized = 0;

    CUptiResult res;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    {
        // For a driver call to launch a kernel:
        if (callbackId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            CUpti_CallbackData const *pData = static_cast<CUpti_CallbackData const *>(pCallbackData);
            CUcontext ctx = pData->context;

            // On entry, enable / update profiling as needed
            if (pData->callbackSite == CUPTI_API_ENTER)
            {
                // Check for this context in the configured contexts
                // If not configured, it isn't compatible with profiling
                ctxDataMutex.lock();
                if (contextData.count(ctx) > 0)
                {
                    // If at maximum number of ranges, end session and reset
                    if (contextData[ctx].curRanges == contextData[ctx].maxNumRanges)
                    {
                        EndSession(contextData[ctx]);
                        contextData[ctx].curRanges = 0;
                    }

                    // If no currently enabled session on this context, start one
                    if (contextData[ctx].curRanges == 0)
                    {
                        InitializeContextData(contextData[ctx]);
                        StartSession(contextData[ctx]);
                    }

                    // Increment curRanges
                    contextData[ctx].curRanges++;
                }
                ctxDataMutex.unlock();
            }
        }
    }
    else if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        // When a context is created, check to see whether the device is compatible with the Profiler API
        if (callbackId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            CUpti_ResourceData const *pResourceData = static_cast<CUpti_ResourceData const *>(pCallbackData);
            CUcontext ctx = pResourceData->context;

            // Configure handler for new context under lock
            CtxProfilerData data = { };

            data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(data.deviceId)));

            RUNTIME_API_CALL(cudaGetDeviceProperties(&(data.deviceProp), data.deviceId));

            // Initialize profiler API and test device compatibility
            InitializeState();
            CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
            params.cuDevice = data.deviceId;
            CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

            // If valid for profiling, set up profiler and save to shared structure
            ctxDataMutex.lock();
            if (params.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
            {
                // Update shared structures
                contextData[ctx] = data;
                InitializeContextData(contextData[ctx]);
            }
            else
            {
                if (contextData.count(ctx))
                {
                    // Update shared structures
                    contextData.erase(ctx);
                }

                cerr << "libinjection: Unable to profile context on device " << data.deviceId << endl;

                if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice architecture is not supported" << endl;
                }

                if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice sli configuration is not supported" << endl;
                }

                if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice vgpu configuration is not supported" << endl;
                }
                else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
                {
                    cerr << "\tdevice vgpu configuration disabled profiling support" << endl;
                }

                if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice confidential compute configuration is not supported" << endl;
                }

                if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
                }

                if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    ::std::cerr << "\tWSL is not supported" << ::std::endl;
                }
            }
            ctxDataMutex.unlock();
        }
    }

    return;
}

// Register callbacks for several points in target application execution
void
RegisterCallbacks()
{
    // One subscriber is used to register multiple callback domains
    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)ProfilerCallbackHandler, NULL));
    // Runtime callback domain is needed for kernel launch callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    // Resource callback domain is needed for context creation callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));

    // Register callback for application exit
    atexit(EndExecution);
}

static bool injectionInitialized = false;

// InitializeInjection will be called by the driver when the tool is loaded
// by CUDA_INJECTION64_PATH
extern "C" DLLEXPORT int
InitializeInjection()
{
    if (injectionInitialized == false)
    {
        injectionInitialized = true;

        // Read in optional list of metrics to gather
        char *pMetricEnv = getenv("INJECTION_METRICS");
        if (pMetricEnv != NULL)
        {
            char * tok = strtok(pMetricEnv, " ;,");
            do
            {
                cout << "Requesting metric '" << tok << "'" << endl;
                metricNames.push_back(string(tok));
                tok = strtok(NULL, " ;,");
            } while (tok != NULL);
        }
        else
        {
            metricNames.push_back("sm__cycles_elapsed.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
        }

        // Subscribe to some callbacks
        RegisterCallbacks();
    }
    return 1;
}
