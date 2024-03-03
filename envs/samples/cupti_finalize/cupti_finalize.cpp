/*
 * Copyright 2020-2022 NVIDIA Corporation. All rights reserved.
 *
 * Sample CUPTI based injection to attach and detach CUPTI
 * For detaching, it uses CUPTI API cuptiFinalize().
 *
 * It is recommended to invoke API cuptiFinalize() in the
 * exit callsite of any CUDA Driver/Runtime API.
 *
 * API cuptiFinalize() destroys and cleans up all the
 * resources associated with CUPTI in the current process.
 * After CUPTI detaches from the process, the process will
 * keep on running with no CUPTI attached to it.
 *
 * CUPTI can be attached by calling any CUPTI API as CUPTI
 * supports lazy initialization. Any subsequent CUPTI API
 * call will reinitialize the CUPTI.
 *
 * You can attach and detach CUPTI any number of times.
 *
 * After building the sample, set the following environment variable
 * export CUDA_INJECTION64_PATH=<full_path>/libCuptiFinalize.so
 * Add CUPTI library in LD_LIBRARY_PATH and run any CUDA sample
 * with runtime more than 10 sec for demonstration of the
 * CUPTI sample
 */

// System headers
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Macros
#define STDCALL

#define PTHREAD_CALL(call)                                                         \
do                                                                                 \
{                                                                                  \
    int _status = call;                                                            \
    if (_status != 0)                                                              \
    {                                                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error code %d.\n",  \
                __FILE__, __LINE__, #call, _status);                               \
                                                                                   \
        exit(EXIT_FAILURE);                                                        \
    }                                                                              \
} while (0)


// Global Structure.
typedef struct InjectionGlobals_st
{
    volatile uint32_t       initialized;
    volatile uint32_t       detachCupti;
    CUpti_SubscriberHandle  subscriberHandle;

    int                     frequency;
    int                     tracingEnabled;
    int                     terminateThread;

    pthread_t               dynamicThread;
    pthread_mutex_t         mutexFinalize;
    pthread_cond_t          mutexCondition;
} InjectionGlobals;

InjectionGlobals injectionGlobals;

// Functions
static void
InitializeInjectionGlobals(void)
{
    injectionGlobals.initialized        = 0;
    injectionGlobals.subscriberHandle   = NULL;
    injectionGlobals.detachCupti        = 0;
    injectionGlobals.frequency          = 2; // in seconds
    injectionGlobals.tracingEnabled     = 0;
    injectionGlobals.terminateThread    = 0;
    injectionGlobals.mutexFinalize      = PTHREAD_MUTEX_INITIALIZER;
    injectionGlobals.mutexCondition     = PTHREAD_COND_INITIALIZER;
}

static void
AtExitHandler(void)
{
    injectionGlobals.terminateThread = 1;

    // Force flush the activity buffers.
    if (injectionGlobals.tracingEnabled)
    {
        CUPTI_API_CALL(cuptiActivityFlushAll(1));
    }

    PTHREAD_CALL(pthread_join(injectionGlobals.dynamicThread, NULL));
}

void RegisterAtExitHandler(void)
{
    atexit(&AtExitHandler);
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

    // This code path is taken only when we wish to perform the CUPTI teardown.
    if (injectionGlobals.detachCupti)
    {
        switch (domain)
        {
            case CUPTI_CB_DOMAIN_RUNTIME_API:
            case CUPTI_CB_DOMAIN_DRIVER_API:
            {
                if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
                {
                    // Detach CUPTI calling cuptiFinalize() API.
                    CUPTI_API_CALL(cuptiFinalize());
                    PTHREAD_CALL(pthread_cond_broadcast(&injectionGlobals.mutexCondition));
                }
                break;
            }
            default:
                break;
        }
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

    // Subscribe Driver and Runtime callbacks to call cuptiFinalize in the entry/exit callback of these APIs.
    CUPTI_API_CALL(cuptiEnableDomain(1, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_API_CALL(cuptiEnableDomain(1, injectionGlobals.subscriberHandle, CUPTI_CB_DOMAIN_DRIVER_API));

    // Enable CUPTI activities.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

void *DynamicAttachDetach(
    void *arg)
{
    while (!injectionGlobals.terminateThread)
    {
        sleep(injectionGlobals.frequency);

        // Check the condition again after sleep.
        if (injectionGlobals.terminateThread)
        {
            break;
        }

        // Turn on/off CUPTI at a regular interval.
        if (injectionGlobals.tracingEnabled)
        {
            printf("\nCUPTI detach starting ...\n");

            // Force flush activity buffers.
            DeInitCuptiTrace();
            injectionGlobals.detachCupti = 1;

            // Lock and wait for callbackHandler() to perform CUPTI teardown.
            PTHREAD_CALL(pthread_mutex_lock(&injectionGlobals.mutexFinalize));
            PTHREAD_CALL(pthread_cond_wait(&injectionGlobals.mutexCondition, &injectionGlobals.mutexFinalize));
            PTHREAD_CALL(pthread_mutex_unlock(&injectionGlobals.mutexFinalize));

            printf("CUPTI detach completed.\n");

            injectionGlobals.detachCupti = 0;
            injectionGlobals.tracingEnabled = 0;
            injectionGlobals.subscriberHandle = 0;

        }
        else
        {
            printf("\nCUPTI attach starting ...\n");

            SetupCupti();
            injectionGlobals.tracingEnabled = 1;

            printf("CUPTI attach completed.\n");
        }
    }

    return NULL;
}

extern "C" int
InitializeInjection(void)
{
    if (injectionGlobals.initialized)
    {
        return 1;
    }

    // Initialize InjectionGlobals structure.
    InitializeInjectionGlobals();

    // Initialize Mutex.
    PTHREAD_CALL(pthread_mutex_init(&injectionGlobals.mutexFinalize, 0));

    RegisterAtExitHandler();

    // Initialize CUPTI.
    SetupCupti();
    injectionGlobals.tracingEnabled = 1;

    // Launch the thread.
    PTHREAD_CALL(pthread_create(&injectionGlobals.dynamicThread, NULL, DynamicAttachDetach, NULL));
    injectionGlobals.initialized = 1;

    return 1;
}
