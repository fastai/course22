/**
 * Copyright 2022 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////

#ifndef HELPER_CUPTI_H_
#define HELPER_CUPTI_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#if defined(WIN32) || defined(_WIN32)
#define stricmp _stricmp
#else
#define stricmp strcasecmp
#endif

#define DRIVER_API_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUresult _status = apiFunctionCall;                                             \
    if (_status != CUDA_SUCCESS)                                                    \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuGetErrorString(_status, &pErrorString);                                   \
                                                                                    \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define RUNTIME_API_CALL(apiFunctionCall)                                           \
do                                                                                  \
{                                                                                   \
    cudaError_t _status = apiFunctionCall;                                          \
    if (_status != cudaSuccess)                                                     \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, cudaGetErrorString(_status)); \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_API_CALL(apiFunctionCall)                                             \
do                                                                                  \
{                                                                                   \
    CUptiResult _status = apiFunctionCall;                                          \
    if (_status != CUPTI_SUCCESS)                                                   \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuptiGetResultString(_status, &pErrorString);                               \
                                                                                    \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_UTIL_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUptiUtilResult _status = apiFunctionCall;                                      \
    if (_status != CUPTI_UTIL_SUCCESS)                                              \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: function %s failed with error %d.\n",        \
                __FILE__, __LINE__, #apiFunctionCall, _status);                     \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define NVPW_API_CALL(apiFunctionCall)                                              \
do                                                                                  \
{                                                                                   \
    NVPA_Status _status = apiFunctionCall;                                          \
    if (_status != NVPA_STATUS_SUCCESS)                                             \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %d.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, _status);                     \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define MEMORY_ALLOCATION_CALL(variable)                                            \
do                                                                                  \
{                                                                                   \
    if (variable == NULL)                                                           \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n",                \
                __FILE__, __LINE__);                                                \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define TEST_ASSERT(testValue)                                                      \
do                                                                                  \
{                                                                                   \
    if (!(testValue))                                                               \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Condition " #testValue " failed.\n",         \
                __FILE__, __LINE__);                                                \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#endif // HELPER_CUPTI_H_