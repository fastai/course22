/*
 * Copyright 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * CUPTI based tracing injection to trace any CUDA application.
 * This sample demonstrates how to use activity
 * and callback APIs in the injection code.
 * Refer to the README.txt file for usage.
 *
 * Workflow in brief:
 *
 *  After the initialization routine returns, the application resumes running,
 *  with the registered callbacks triggering as expected.
 *  Subscribed to ProfilerStart and ProfilerStop callbacks. These callbacks
 *  control the collection of profiling data.
 *
 *  ProfilerStart callback:
 *      Start the collection by enabling activities. Also enable callback for
 *      the API cudaDeviceReset to flush activity buffers.
 *
 *  ProfilerStop callback:
 *      Get all the activity buffers which have all the activity records completed
 *      by using cuptiActivityFlushAll() API and then disable cudaDeviceReset callback
 *      and all the activities to stop collection.
 *
 *  AtExitHandler:
 *      Register to the atexit handler to get all the activity buffers including the ones
 *      which have incomplete activity records by using force flush API
 *      cuptiActivityFlushAll(1).
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Detours for Windows
#ifdef _WIN32
#include "detours.h"
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

// Macros
#define IS_ACTIVITY_SELECTED(activitySelect, activityKind)                               \
    (activitySelect & (1LL << activityKind))

#define SELECT_ACTIVITY(activitySelect, activityKind)                                    \
    (activitySelect |= (1LL << activityKind))

// Variable related to initialize injection.
std::mutex initializeInjectionMutex;

// Global Structure
typedef struct InjectionGlobals_st
{
    volatile uint32_t       initialized;
    CUpti_SubscriberHandle  subscriberHandle;
    int                     tracingEnabled;
    uint64_t                profileMode;
} InjectionGlobals;

InjectionGlobals injectionGlobals;

// Functions
static void
InitializeInjectionGlobals(void)
{
    injectionGlobals.initialized        = 0;
    injectionGlobals.subscriberHandle   = NULL;
    injectionGlobals.tracingEnabled     = 0;
    injectionGlobals.profileMode        = 0;
}

static void
AtExitHandler(void)
{
    CUPTI_API_CALL(cuptiGetLastError());

    // Force flush the activity buffers.
    if (injectionGlobals.tracingEnabled)
    {
        CUPTI_API_CALL(cuptiActivityFlushAll(1));
    }
}

#ifdef _WIN32
typedef void(WINAPI *rtlExitUserProcess_t)(uint32_t exitCode);
rtlExitUserProcess_t Real_RtlExitUserProcess = NULL;

// Detour_RtlExitUserProcess.
void WINAPI
Detour_RtlExitUserProcess(
    uint32_t exitCode)
{
    AtExitHandler();

    Real_RtlExitUserProcess(exitCode);
}
#endif

void
RegisterAtExitHandler(void)
{
#ifdef _WIN32
    {
        // It's unsafe to use atexit(), static destructors, DllMain PROCESS_DETACH, etc.
        // because there's no way to guarantee the CUDA driver is still in a valid state
        // when you get to those, due to the undefined order of dynamic library tear-down
        // during process destruction.
        // Also, the first thing the Windows kernel does when any thread in a process
        // calls exit() is to immediately terminate all other threads, without any kind of
        // synchronization.
        // So the only valid time to do any in-process cleanup at exit() is before control
        // is passed to the kernel. Use Detours to intercept a low-level ntdll.dll
        // function "RtlExitUserProcess".
        int detourStatus = 0;
        FARPROC proc;

        // ntdll.dll will always be loaded, no need to load the library.
        HMODULE ntDll = GetModuleHandle(TEXT("ntdll.dll"));
        if (!ntDll)
        {
            detourStatus = 1;
            goto DetourError;
        }

        proc = GetProcAddress(ntDll, "RtlExitUserProcess");
        if (!proc)
        {
            detourStatus = 1;
            goto DetourError;
        }
        Real_RtlExitUserProcess = (rtlExitUserProcess_t)proc;

        // Begin a detour transaction
        if (DetourTransactionBegin() != ERROR_SUCCESS)
        {
            detourStatus = 1;
            goto DetourError;
        }

        if (DetourUpdateThread(GetCurrentThread()) != ERROR_SUCCESS)
        {
            detourStatus = 1;
            goto DetourError;
        }

        DetourSetIgnoreTooSmall(TRUE);

        if (DetourAttach((void **)&Real_RtlExitUserProcess,
                         (void *)Detour_RtlExitUserProcess) != ERROR_SUCCESS)
        {
            detourStatus = 1;
            goto DetourError;
        }

        // Commit the transaction
        if (DetourTransactionCommit() != ERROR_SUCCESS)
        {
            detourStatus = 1;
            goto DetourError;
        }
    DetourError:
        if (detourStatus != 0)
        {
            atexit(&AtExitHandler);
        }
    }
#else
    atexit(&AtExitHandler);
#endif
}

static CUptiResult
SelectActivities()
{
    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_DRIVER);
    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_RUNTIME);
    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_MEMSET);
    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_MEMCPY);

    return CUPTI_SUCCESS;
}

static CUptiResult
EnableCuptiActivities(
    CUcontext context)
{
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_API_CALL(cuptiEnableCallback(1, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));

    CUPTI_API_CALL(SelectActivities());

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i)
    {
        if (IS_ACTIVITY_SELECTED(injectionGlobals.profileMode, i))
        {
            // If context is NULL activities are being enabled after CUDA initialization.
            // Else the activities are being enabled on cudaProfilerStart API.
            if (context == NULL)
            {
                CUPTI_API_CALL(cuptiActivityEnable((CUpti_ActivityKind)i));
            }
            else
            {
                // Since some activities are not supported at context mode,
                // enable them in global mode if context mode fails.
                result = cuptiActivityEnableContext(context, (CUpti_ActivityKind)i);

                if (result == CUPTI_ERROR_INVALID_KIND)
                {
                    cuptiGetLastError();
                    result = cuptiActivityEnable((CUpti_ActivityKind)i);
                }
                else if (result != CUPTI_SUCCESS)
                {
                    CUPTI_API_CALL(result);
                }
            }
        }
    }

    return result;
}

static CUptiResult
DisableCuptiActivities(
    CUcontext ctx)
{
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_API_CALL(cuptiEnableCallback(0, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i)
    {
        if (IS_ACTIVITY_SELECTED(injectionGlobals.profileMode, i))
        {
            // Since some activities are not supported at context mode,
            // disable them in global mode if context mode fails.
            result = cuptiActivityDisableContext(ctx, (CUpti_ActivityKind)i);

            if (result == CUPTI_ERROR_INVALID_KIND)
            {
                cuptiGetLastError();
                CUPTI_API_CALL(cuptiActivityDisable((CUpti_ActivityKind)i));
            }
            else if (result != CUPTI_SUCCESS)
            {
                CUPTI_API_CALL(result);
            }
        }
    }

    return CUPTI_SUCCESS;
}

static CUptiResult
OnCudaDeviceReset(void)
{
    // Flush all activity buffers.
    CUPTI_API_CALL(cuptiActivityFlushAll(0));

    return CUPTI_SUCCESS;
}

static CUptiResult
OnProfilerStart(
    CUcontext context)
{
    if (context == NULL)
    {
        // Don't do anything if context is NULL.
        return CUPTI_SUCCESS;
    }

    CUPTI_API_CALL(EnableCuptiActivities(context));

    return CUPTI_SUCCESS;
}

static CUptiResult
OnProfilerStop(
    CUcontext context)
{
    if (context == NULL)
    {
        // Don't do anything if context is NULL.
        return CUPTI_SUCCESS;
    }

    CUPTI_API_CALL(cuptiActivityFlushAll(0));
    CUPTI_API_CALL(DisableCuptiActivities(context));

    return CUPTI_SUCCESS;
}

void CUPTIAPI
InjectionCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData)
{
    const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;

    // Check last error.
    CUPTI_API_CALL(cuptiGetLastError());

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {
            switch (callbackId)
            {
                case CUPTI_DRIVER_TRACE_CBID_cuProfilerStart:
                {
                    // We start profiling collection on exit of the API.
                    if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
                    {
                        OnProfilerStart(pCallbackInfo->context);
                    }
                    break;
                }
                case CUPTI_DRIVER_TRACE_CBID_cuProfilerStop:
                {
                    // We stop profiling collection on entry of the API.
                    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
                    {
                        OnProfilerStop(pCallbackInfo->context);
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
            switch (callbackId)
            {
                case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
                {
                    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
                    {
                        CUPTI_API_CALL(OnCudaDeviceReset());
                    }
                    break;
                }
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
}

static void
SetupCupti(void)
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, (void *)InjectionCallbackHandler, stdout);

    injectionGlobals.subscriberHandle = globals.subscriberHandle;

    // Subscribe Driver callback to call OnProfilerStart/OnProfilerStop function.
    CUPTI_API_CALL(cuptiEnableCallback(1, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuProfilerStart));
    CUPTI_API_CALL(cuptiEnableCallback(1, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuProfilerStop));

    // Enable CUPTI activities.
    CUPTI_API_CALL(EnableCuptiActivities(NULL));
}

#ifdef _WIN32
extern "C" __declspec(dllexport) int
InitializeInjection(void)
#else
extern "C" int
InitializeInjection(void)
#endif
{
    if (injectionGlobals.initialized)
    {
        return 1;
    }

    initializeInjectionMutex.lock();

    // Initialize injection global options.
    InitializeInjectionGlobals();

    RegisterAtExitHandler();

    // Initialize CUPTI.
    SetupCupti();

    injectionGlobals.tracingEnabled = 1;
    injectionGlobals.initialized = 1;

    initializeInjectionMutex.unlock();

    return 1;
}
