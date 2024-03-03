#ifndef NVPERF_CUDA_HOST_H
#define NVPERF_CUDA_HOST_H

/*
 * Copyright 2014-2022  NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
 * of a form of NVIDIA software license agreement.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stddef.h>
#include <stdint.h>
#include "nvperf_common.h"
#include "nvperf_host.h"

#if defined(__GNUC__) && defined(NVPA_SHARED_LIB)
    #pragma GCC visibility push(default)
    #if !defined(NVPW_LOCAL)
        #define NVPW_LOCAL __attribute__ ((visibility ("hidden")))
    #endif
#else
    #if !defined(NVPW_LOCAL)
        #define NVPW_LOCAL
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file   nvperf_cuda_host.h
 */

    /// 'NVPA_MetricsContext' and its APIs are deprecated, please use 'NVPW_MetricsEvaluator' and its APIs instead.
    typedef struct NVPA_MetricsContext NVPA_MetricsContext;

    typedef struct NVPW_CUDA_MetricsContext_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const char* pChipName;
        /// [out]
        struct NVPA_MetricsContext* pMetricsContext;
    } NVPW_CUDA_MetricsContext_Create_Params;
#define NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsContext_Create_Params, pMetricsContext)

    NVPA_Status NVPW_CUDA_MetricsContext_Create(NVPW_CUDA_MetricsContext_Create_Params* pParams);

    typedef struct NVPW_CUDA_RawMetricsConfig_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        NVPA_ActivityKind activityKind;
        /// [in]
        const char* pChipName;
        /// [out] new NVPA_RawMetricsConfig object
        struct NVPA_RawMetricsConfig* pRawMetricsConfig;
    } NVPW_CUDA_RawMetricsConfig_Create_Params;
#define NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_RawMetricsConfig_Create_Params, pRawMetricsConfig)

    NVPA_Status NVPW_CUDA_RawMetricsConfig_Create(NVPW_CUDA_RawMetricsConfig_Create_Params* pParams);

    typedef struct NVPW_CUDA_RawMetricsConfig_Create_V2_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        NVPA_ActivityKind activityKind;
        /// [in] accepted for chips supported at the time-of-release.
        const char* pChipName;
        /// [in] buffer with counter availability image - required for future chip support
        const uint8_t* pCounterAvailabilityImage;
        /// [out] new NVPA_RawMetricsConfig object
        struct NVPA_RawMetricsConfig* pRawMetricsConfig;
    } NVPW_CUDA_RawMetricsConfig_Create_V2_Params;
#define NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_RawMetricsConfig_Create_V2_Params, pRawMetricsConfig)

    /// Use either 'pChipName' or 'pCounterAvailabilityImage'.
    NVPA_Status NVPW_CUDA_RawMetricsConfig_Create_V2(NVPW_CUDA_RawMetricsConfig_Create_V2_Params* pParams);

    typedef struct NVPW_CUDA_CounterDataBuilder_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] accepted for chips supported at the time-of-release.
        const char* pChipName;
        /// [in] buffer with counter availability image - required for future chip support
        const uint8_t* pCounterAvailabilityImage;
        /// [out] new NVPA_CounterDataBuilder object
        struct NVPA_CounterDataBuilder* pCounterDataBuilder;
    } NVPW_CUDA_CounterDataBuilder_Create_Params;
#define NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_CounterDataBuilder_Create_Params, pCounterDataBuilder)

    /// Use either 'pChipName' or 'pCounterAvailabilityImage'.
    NVPA_Status NVPW_CUDA_CounterDataBuilder_Create(NVPW_CUDA_CounterDataBuilder_Create_Params* pParams);

    typedef struct NVPW_MetricsEvaluator NVPW_MetricsEvaluator;

    typedef struct NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] accepted for chips supported at the time-of-release.
        const char* pChipName;
        /// [in] buffer with counter availability image - required for future chip support
        const uint8_t* pCounterAvailabilityImage;
        /// [out]
        size_t scratchBufferSize;
    } NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params;
#define NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params, scratchBufferSize)

    /// Use either 'pChipName' or 'pCounterAvailabilityImage'.
    NVPA_Status NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params* pParams);

    typedef struct NVPW_CUDA_MetricsEvaluator_Initialize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        uint8_t* pScratchBuffer;
        /// [in] the size of the 'pScratchBuffer' array, should be at least the size of the 'scratchBufferSize' returned
        /// by 'NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize'
        size_t scratchBufferSize;
        /// [in] accepted for chips supported at the time-of-release.
        const char* pChipName;
        /// [in] buffer with counter availability image - required for future chip support
        const uint8_t* pCounterAvailabilityImage;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in] must be provided if 'pCounterDataImage' is not NULL
        size_t counterDataImageSize;
        /// [out]
        struct NVPW_MetricsEvaluator* pMetricsEvaluator;
    } NVPW_CUDA_MetricsEvaluator_Initialize_Params;
#define NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsEvaluator_Initialize_Params, pMetricsEvaluator)

    /// Use one of 'pChipName', 'pCounterAvailabilityImage', or 'pCounterDataImage'. 'pChipName' or
    /// 'pCounterAvailabilityImage' will create a metrics evaluator based on a virtual device while 'pCounterDataImage'
    /// will create a metrics evaluator based on the actual device.
    NVPA_Status NVPW_CUDA_MetricsEvaluator_Initialize(NVPW_CUDA_MetricsEvaluator_Initialize_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(NVPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // NVPERF_CUDA_HOST_H
