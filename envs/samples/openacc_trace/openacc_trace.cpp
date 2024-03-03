/*
 * Copyright 2015-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI library for OpenACC data collection.
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA headers
#include <cuda.h>

// OpenACC headers
#include <openacc.h>

// CUPTI headers
#include "helper_cupti_activity.h"

static size_t s_OpenAccRecords = 0;

static void
OpenAccActivityRecords(
    CUpti_Activity *pRecord)
{
    switch (pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
        case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
        case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
        {
            CUpti_ActivityOpenAcc *pOpenAcc = (CUpti_ActivityOpenAcc *)pRecord;

            if (pOpenAcc->deviceType != acc_device_nvidia)
            {
                printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", pOpenAcc->deviceType, acc_device_nvidia);
                exit(EXIT_FAILURE);
            }

            s_OpenAccRecords++;
        }
            break;
        default:
            break;
    }
}

static void
AtExitHandler()
{
    DeInitCuptiTrace();
    printf("Found %llu OpenACC records\n", (long long unsigned) s_OpenAccRecords);
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData*)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = OpenAccActivityRecords;
    pUserData->printActivityRecords        = 1;
    // Set flag to 1 to avoid CUPTI subscription from happening.
    pUserData->skipCuptiSubscription       = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, NULL, stdout);

    // Enable OpenACC activity kinds.
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER));

    printf("Initialized CUPTI support for OpenACC.\n");
}

// acc_register_library is defined by the OpenACC tools interface
// and allows to register this library with the OpenACC runtime.

extern "C" void
acc_register_library(
    void *pProfilerRegister,
    void *pProfilerUnregister,
    void *pProfilerrofLookup)
{
    // Once connected to the OpenACC runtime, initialize CUPTI's OpenACC interface.
    if (cuptiOpenACCInitialize(pProfilerRegister, pProfilerUnregister, pProfilerrofLookup) != CUPTI_SUCCESS)
    {
        printf("Error: Failed to initialize CUPTI support for OpenACC.\n");
        exit(EXIT_FAILURE);
    }

    SetupCupti();

    // At program exit, flush CUPTI buffers and print results.
    atexit(AtExitHandler);
}
