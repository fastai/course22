/*
 * Copyright 2011-2020   NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_PROFILER_TARGET_H_)
#define _CUPTI_PROFILER_TARGET_H_

#include <cuda.h>
#include <cupti_result.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_PROFILER_API CUPTI Profiling API
 * Functions, types, and enums that implement the CUPTI Profiling API.
 * @{
 */
#ifndef CUPTI_PROFILER_STRUCT_SIZE
#define CUPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

/**
 * \brief Profiler range attribute
 *
 * A metric enabled in the session's configuration is collected separately per unique range-stack in the pass.
 * This is an attribute to collect metrics around each kernel in a profiling session or in an user defined range.
 */
typedef enum
{
    /**
     * Invalid value
     */
    CUPTI_Range_INVALID,
    /**
     * Ranges are auto defined around each kernel in a profiling session
     */
    CUPTI_AutoRange,
    /**
     * A range in which metric data to be collected is defined by the user
     */
    CUPTI_UserRange,
    /**
     * Range count
     */
    CUPTI_Range_COUNT,
} CUpti_ProfilerRange;

/**
 * \brief Profiler replay attribute
 *
 * For metrics which require multipass collection, a replay of the GPU kernel(s) is required.
 * This is an attribute which specify how the replay of the kernel(s) to be measured is done.
 */
typedef enum
{
    /**
     * Invalid Value
     */
    CUPTI_Replay_INVALID,
    /**
     * Replay is done by CUPTI user around the process
     */
    CUPTI_ApplicationReplay,
    /**
     * Replay is done around kernel implicitly by CUPTI
     */
    CUPTI_KernelReplay,
    /**
     * Replay is done by CUPTI user within a process
     */
    CUPTI_UserReplay,
    /**
     * Replay count
     */
    CUPTI_Replay_COUNT,
} CUpti_ProfilerReplayMode;

/**
 * \brief Default parameter for cuptiProfilerInitialize
 */
typedef struct CUpti_Profiler_Initialize_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_Initialize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

} CUpti_Profiler_Initialize_Params;
#define CUpti_Profiler_Initialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Initialize_Params, pPriv)

/**
 * \brief Default parameter for cuptiProfilerDeInitialize
 */
typedef struct CUpti_Profiler_DeInitialize_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

} CUpti_Profiler_DeInitialize_Params;
#define CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_DeInitialize_Params, pPriv)

/**
 * \brief Initializes the profiler interface
 *
 * Loads the required libraries in the process address space.
 * Sets up the hooks with the CUDA driver.
 */
CUptiResult CUPTIAPI cuptiProfilerInitialize(CUpti_Profiler_Initialize_Params *pParams);

/**
 * \brief DeInitializes the profiler interface
 */
CUptiResult CUPTIAPI cuptiProfilerDeInitialize(CUpti_Profiler_DeInitialize_Params *pParams);

/**
 * \brief Input parameter to define the counterDataImage
 */
typedef struct CUpti_Profiler_CounterDataImageOptions
{
    size_t structSize;                                          //!< [in] CUpti_Profiler_CounterDataImageOptions_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    const uint8_t* pCounterDataPrefix;                          /**< [in] Address of CounterDataPrefix generated from NVPW_CounterDataBuilder_GetCounterDataPrefix().
                                                                    Must be align(8).*/
    size_t counterDataPrefixSize;                               //!< [in] Size of CounterDataPrefix generated from NVPW_CounterDataBuilder_GetCounterDataPrefix().
    uint32_t maxNumRanges;                                      //!< [in] Maximum number of ranges that can be profiled
    uint32_t maxNumRangeTreeNodes;                              //!< [in] Maximum number of RangeTree nodes; must be >= maxNumRanges
    uint32_t maxRangeNameLength;                                //!< [in] Maximum string length of each RangeName, including the trailing NULL character
} CUpti_Profiler_CounterDataImageOptions;
#define CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE                       CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_CounterDataImageOptions, maxRangeNameLength)

/**
 * \brief Params for cuptiProfilerCounterDataImageCalculateSize
 */
typedef struct CUpti_Profiler_CounterDataImage_CalculateSize_Params
{
    size_t structSize;                                          //!< [in] CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    size_t sizeofCounterDataImageOptions;                       //!< [in] CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE
    const CUpti_Profiler_CounterDataImageOptions* pOptions;     //!< [in] Pointer to Counter Data Image Options
    size_t counterDataImageSize;                                //!< [out]
} CUpti_Profiler_CounterDataImage_CalculateSize_Params;
#define CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE         CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_CounterDataImage_CalculateSize_Params, counterDataImageSize)

/**
 * \brief Params for cuptiProfilerCounterDataImageInitialize
 */
typedef struct CUpti_Profiler_CounterDataImage_Initialize_Params
{
    size_t structSize;                                          //!< [in] CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    size_t sizeofCounterDataImageOptions;                       //!< [in] CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE
    const CUpti_Profiler_CounterDataImageOptions* pOptions;     //!< [in] Pointer to Counter Data Image Options
    size_t counterDataImageSize;                                //!< [in] Size calculated from cuptiProfilerCounterDataImageCalculateSize
    uint8_t* pCounterDataImage;                                 //!< [in] The buffer to be initialized.
} CUpti_Profiler_CounterDataImage_Initialize_Params;
#define CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE            CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_CounterDataImage_Initialize_Params, pCounterDataImage)

/**
 * \brief A CounterData image allocates space for values for each counter for each range.
 *
 * User borne the resposibility of managing the counterDataImage allocations.
 * CounterDataPrefix contains meta data about the metrics that will be stored in counterDataImage.
 * Use these APIs to calculate the allocation size and initialize counterData image.
 */
CUptiResult cuptiProfilerCounterDataImageCalculateSize(CUpti_Profiler_CounterDataImage_CalculateSize_Params* pParams);
CUptiResult cuptiProfilerCounterDataImageInitialize(CUpti_Profiler_CounterDataImage_Initialize_Params* pParams);

/**
 * \brief Params for cuptiProfilerCounterDataImageCalculateScratchBufferSize
 */
typedef struct CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    size_t counterDataImageSize;                            //!< [in] size calculated from cuptiProfilerCounterDataImageCalculateSize
    uint8_t* pCounterDataImage;                             //!< [in]
    size_t counterDataScratchBufferSize;                    //!< [out]
} CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params;
#define CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE    CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params, counterDataScratchBufferSize)

/**
 * \brief Params for cuptiProfilerCounterDataImageInitializeScratchBuffer
 */
typedef struct CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    size_t counterDataImageSize;                            //!< [in] size calculated from cuptiProfilerCounterDataImageCalculateSize
    uint8_t* pCounterDataImage;                             //!< [in]
    size_t counterDataScratchBufferSize;                    //!< [in] size calculated using cuptiProfilerCounterDataImageCalculateScratchBufferSize
    uint8_t* pCounterDataScratchBuffer;                     //!< [in] the scratch buffer to be initialized.
} CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params;
#define CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE       CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params, pCounterDataScratchBuffer)

/**
 * \brief A temporary storage for CounterData image needed for internal operations
 *
 * Use these APIs to calculate the allocation size and initialize counterData image scratch buffer.
 */
CUptiResult cuptiProfilerCounterDataImageCalculateScratchBufferSize(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* pParams);
CUptiResult cuptiProfilerCounterDataImageInitializeScratchBuffer(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);

/**
 * \brief Params for cuptiProfilerBeginSession
 */
typedef struct CUpti_Profiler_BeginSession_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_BeginSession_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    size_t counterDataImageSize;                            //!< [in] size calculated from cuptiProfilerCounterDataImageCalculateSize
    uint8_t* pCounterDataImage;                             //!< [in] address of CounterDataImage
    size_t counterDataScratchBufferSize;                    //!< [in] size calculated from cuptiProfilerCounterDataImageInitializeScratchBuffer
    uint8_t* pCounterDataScratchBuffer;                     //!< [in] address of CounterDataImage scratch buffer
    uint8_t bDumpCounterDataInFile;                          //!< [in] [optional]
    const char* pCounterDataFilePath;                        //!< [in] [optional]
    CUpti_ProfilerRange range;                               //!< [in] CUpti_ProfilerRange
    CUpti_ProfilerReplayMode replayMode;                     //!< [in] CUpti_ProfilerReplayMode
    /* Replay options, required when replay is done by cupti user */
    size_t maxRangesPerPass;                                //!< [in] Maximum number of ranges that can be recorded in a single pass.
    size_t maxLaunchesPerPass;                              //!< [in] Maximum number of kernel launches that can be recorded in a single pass; must be >= maxRangesPerPass.

} CUpti_Profiler_BeginSession_Params;
#define CUpti_Profiler_BeginSession_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_BeginSession_Params, maxLaunchesPerPass)
/**
 * \brief Params for cuptiProfilerEndSession
 */
typedef struct CUpti_Profiler_EndSession_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_EndSession_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_EndSession_Params;
#define CUpti_Profiler_EndSession_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_EndSession_Params, ctx)

/**
 * \brief Begin profiling session sets up the profiling on the device
 *
 * Although, it doesn't start the profiling but GPU resources needed for profiling are allocated.
 * Outside of a session, the GPU will return to its normal operating state.
 */
CUptiResult CUPTIAPI cuptiProfilerBeginSession(CUpti_Profiler_BeginSession_Params* pParams);
/**
 * \brief Ends profiling session
 *
 * Frees up the GPU resources acquired for profiling.
 * Outside of a session, the GPU will return to it's normal operating state.
 */
CUptiResult CUPTIAPI cuptiProfilerEndSession(CUpti_Profiler_EndSession_Params* pParams);

/**
 * \brief Params for cuptiProfilerSetConfig
 */
typedef struct CUpti_Profiler_SetConfig_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_SetConfig_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    const uint8_t* pConfig;                                 //!< [in] Config created by NVPW_RawMetricsConfig_GetConfigImage(). Must be align(8).
    size_t configSize;                                      //!< [in] size of config
    uint16_t minNestingLevel;                               //!< [in] the lowest nesting level to be profiled; must be >= 1
    uint16_t numNestingLevels;                              //!< [in] the number of nesting levels to profile; must be >= 1
    size_t passIndex;                                       //!< [in] Set this to zero for in-app replay; set this to the output of EndPass() for application replay
    uint16_t targetNestingLevel;                            //!< [in] Set this to minNestingLevel for in-app replay; set this to the output of EndPass() for application
} CUpti_Profiler_SetConfig_Params;

#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE                    CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_SetConfig_Params, targetNestingLevel)

/**
 * \brief Params for cuptiProfilerUnsetConfig
 */
typedef struct CUpti_Profiler_UnsetConfig_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_UnsetConfig_Params;
#define CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_UnsetConfig_Params, ctx)

/**
 * \brief Set metrics configuration to be profiled
 *
 * Use these APIs to set the config to profile in a session. It can be used for advanced cases such as where multiple
 * configurations are collected into a single CounterData Image on the need basis, without restarting the session.
 */
CUptiResult CUPTIAPI cuptiProfilerSetConfig(CUpti_Profiler_SetConfig_Params* pParams);
/**
 * \brief Unset metrics configuration profiled
 *
 */
CUptiResult CUPTIAPI cuptiProfilerUnsetConfig(CUpti_Profiler_UnsetConfig_Params* pParams);

/**
 * \brief Params for cuptiProfilerBeginPass
 */
typedef struct CUpti_Profiler_BeginPass_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_BeginPass_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_BeginPass_Params;
#define CUpti_Profiler_BeginPass_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_BeginPass_Params, ctx)

/**
 * \brief Params for cuptiProfilerEndPass
 */
typedef struct CUpti_Profiler_EndPass_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_EndPass_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    uint16_t targetNestingLevel;                            //!  [out] The targetNestingLevel that will be collected by the *next* BeginPass.
    size_t passIndex;                                       //!< [out] The passIndex that will be collected by the *next* BeginPass
    uint8_t allPassesSubmitted;                             //!< [out] becomes true when the last pass has been queued to the GPU
} CUpti_Profiler_EndPass_Params;
#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE                    CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_EndPass_Params, allPassesSubmitted)

/**
 * \brief Replay API: used for multipass collection.

 * These APIs are used if user chooses to replay by itself \ref CUPTI_UserReplay or \ref CUPTI_ApplicationReplay
 * for multipass collection of the metrics configurations.
 * It's a no-op in case of \ref CUPTI_KernelReplay.
 */
CUptiResult cuptiProfilerBeginPass(CUpti_Profiler_BeginPass_Params* pParams);

/**
 * \brief Replay API: used for multipass collection.

 * These APIs are used if user chooses to replay by itself \ref CUPTI_UserReplay or \ref CUPTI_ApplicationReplay
 * for multipass collection of the metrics configurations.
 * Its a no-op in case of \ref CUPTI_KernelReplay.
 * Returns information for next pass.
 */
CUptiResult cuptiProfilerEndPass(CUpti_Profiler_EndPass_Params* pParams);

/**
 * \brief Params for cuptiProfilerEnableProfiling
 */
typedef struct CUpti_Profiler_EnableProfiling_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_EnableProfiling_Params;
#define CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_EnableProfiling_Params, ctx)

/**
 * \brief Params for cuptiProfilerDisableProfiling
 */
typedef struct CUpti_Profiler_DisableProfiling_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_DisableProfiling_Params;
#define CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_DisableProfiling_Params, ctx)

/**
 * \brief Enables Profiling
 *
 * In \ref CUPTI_AutoRange, these APIs are used to enable/disable profiling for the kernels to be executed in
 * a profiling session.
 */
CUptiResult CUPTIAPI cuptiProfilerEnableProfiling(CUpti_Profiler_EnableProfiling_Params* pParams);

/**
 * \brief Disable Profiling
 *
 * In \ref CUPTI_AutoRange, these APIs are used to enable/disable profiling for the kernels to be executed in
 * a profiling session.
 */
CUptiResult CUPTIAPI cuptiProfilerDisableProfiling(CUpti_Profiler_DisableProfiling_Params* pParams);

/**
 * \brief Params for cuptiProfilerIsPassCollected
 */
typedef struct CUpti_Profiler_IsPassCollected_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_IsPassCollected_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    size_t numRangesDropped;                                //!< [out] number of ranges whose data was dropped in the processed pass
    size_t numTraceBytesDropped;                            //!< [out] number of bytes not written to TraceBuffer due to buffer full
    uint8_t onePassCollected;                               //!< [out] true if a pass was successfully decoded
    uint8_t allPassesCollected;                             //!< [out] becomes true when the last pass has been decoded
} CUpti_Profiler_IsPassCollected_Params;
#define CUpti_Profiler_IsPassCollected_Params_STRUCT_SIZE            CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_IsPassCollected_Params, allPassesCollected)

/**
 * \brief Asynchronous call to query if the submitted pass to GPU is collected
 *
 */
CUptiResult CUPTIAPI cuptiProfilerIsPassCollected(CUpti_Profiler_IsPassCollected_Params* pParams);

/**
 * \brief Params for cuptiProfilerFlushCounterData
 */
typedef struct CUpti_Profiler_FlushCounterData_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    size_t numRangesDropped;                                //!< [out] number of ranges whose data was dropped in the processed passes
    size_t numTraceBytesDropped;                            //!< [out] number of bytes not written to TraceBuffer due to buffer full
} CUpti_Profiler_FlushCounterData_Params;
#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE           CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_FlushCounterData_Params, numTraceBytesDropped)

/**
 * \brief Decode all the submitted passes
 *
 * Flush Counter data API to ensure every pass is decoded into the counterDataImage passed at beginSession.
 * This will cause the CPU/GPU sync to collect all the undecoded pass.
 */
CUptiResult CUPTIAPI cuptiProfilerFlushCounterData(CUpti_Profiler_FlushCounterData_Params* pParams);

typedef struct CUpti_Profiler_PushRange_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_PushRange_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
    const char* pRangeName;                                 //!< [in] specifies the range for subsequent launches; must not be NULL
    size_t rangeNameLength;                                 //!< [in] assign to strlen(pRangeName) if known; if set to zero, the library will call strlen()
} CUpti_Profiler_PushRange_Params;
#define CUpti_Profiler_PushRange_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_PushRange_Params, rangeNameLength)

typedef struct CUpti_Profiler_PopRange_Params
{
    size_t structSize;                                      //!< [in] CUpti_Profiler_PopRange_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    CUcontext ctx;                                          //!< [in] if NULL, the current CUcontext is used
} CUpti_Profiler_PopRange_Params;
#define CUpti_Profiler_PopRange_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_PopRange_Params, ctx)


/**
 * \brief Range API's : Push user range
 *
 * Counter data is collected per unique range-stack. Identified by a string label passsed by the user.
 * It's an invalid operation in case of \ref CUPTI_AutoRange.
 */
CUptiResult CUPTIAPI cuptiProfilerPushRange(CUpti_Profiler_PushRange_Params *pParams);

/**
 * \brief Range API's : Pop user range
 *
 * Counter data is collected per unique range-stack. Identified by a string label passsed by the user.
 * It's an invalid operation in case of \ref CUPTI_AutoRange.
 */
CUptiResult CUPTIAPI cuptiProfilerPopRange(CUpti_Profiler_PopRange_Params *pParams);

/**
 * \brief Params for cuptiProfilerGetCounterAvailability
 */
typedef struct CUpti_Profiler_GetCounterAvailability_Params
{
    size_t structSize;                                  //!< [in] CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE
    void* pPriv;                                        //!< [in] assign to NULL
    CUcontext ctx;                                      //!< [in] if NULL, the current CUcontext is used
    size_t counterAvailabilityImageSize;                //!< [in/out] If `pCounterAvailabilityImage` is NULL, then the required size is returned in
                                                        //!< `counterAvailabilityImageSize`, otherwise `counterAvailabilityImageSize` should be set to the size of
                                                        //!< `pCounterAvailabilityImage`, and on return it would be overwritten with number of actual bytes copied
    uint8_t* pCounterAvailabilityImage;                 //!< [in] buffer receiving counter availability image, may be NULL
} CUpti_Profiler_GetCounterAvailability_Params;
#define CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_GetCounterAvailability_Params, pCounterAvailabilityImage)

/**
 * \brief Query counter availibility
 *
 * Use this API to query counter availability information in a buffer which can be used to filter unavailable raw metrics on host.
 * Note: This API may fail, if any profiling or sampling session is active on the specified context or its device.
 */
CUptiResult CUPTIAPI cuptiProfilerGetCounterAvailability(CUpti_Profiler_GetCounterAvailability_Params *pParams);

/// Generic support level enum for CUPTI
typedef enum
{
    CUPTI_PROFILER_CONFIGURATION_UNKNOWN = 0, //!< Configuration support level unknown - either detection code errored out before setting this value, or unable to determine it
    CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED, //!< Profiling is unavailable.  For specific feature fields, this means that the current configuration of this feature does not work with profiling.  For instance, SLI-enabled devices do not support profiling, and this value would be returned for SLI on an SLI-enabled device.
    CUPTI_PROFILER_CONFIGURATION_DISABLED,    //!< Profiling would be available for this configuration, but was disabled by the system
    CUPTI_PROFILER_CONFIGURATION_SUPPORTED    //!< Profiling is supported.  For specific feature fields, this means that the current configuration of this feature works with profiling.  For instance, SLI-enabled devices do not support profiling, and this value would only be returned for devices which are not SLI-enabled.
} CUpti_Profiler_Support_Level;

/**
 * \brief Params for cuptiProfilerDeviceSupported
 */
typedef struct
{
    size_t structSize;                                //!< [in] Must be CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE
    void *pPriv;                                      //!< [in] assign to NULL
    CUdevice cuDevice;                                //!< [in] if NULL, the current CUcontext is used

    CUpti_Profiler_Support_Level isSupported;         //!< [out] overall SUPPORTED / UNSUPPORTED flag representing whether Profiling and PC Sampling APIs work on the given device and configuration. SUPPORTED if all following flags are SUPPORTED, UNSUPPORTED otherwise.

    CUpti_Profiler_Support_Level architecture;        //!< [out] SUPPORTED if the device architecture level supports the Profiling API (Compute Capability >= 7.0), UNSUPPORTED otherwise
    CUpti_Profiler_Support_Level sli;                 //!< [out] SUPPORTED if SLI is not enabled, UNSUPPORTED otherwise
    CUpti_Profiler_Support_Level vGpu;                //!< [out] SUPPORTED if vGPU is supported and profiling is enabled, DISABLED if profiling is supported but not enabled, UNSUPPORTED otherwise
    CUpti_Profiler_Support_Level confidentialCompute; //!< [out] SUPPORTED if confidential compute is not enabled, UNSUPPORTED otherwise
    CUpti_Profiler_Support_Level cmp;                 //!< [out] SUPPORTED if not NVIDIA Crypto Mining Processors (CMP), UNSUPPORTED otherwise
    CUpti_Profiler_Support_Level wsl;                 //!< [out] SUPPORTED if WSL supported, UNSUPPORTED otherwise
} CUpti_Profiler_DeviceSupported_Params;
#define CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_DeviceSupported_Params, confidentialCompute)

/**
 * \brief Query device compatibility with Profiling API
 *
 * Use this call to determine whether a compute device and configuration are compatible with the Profiling API.
 * If the configuration does not support profiling, one of several flags will indicate why.
 */
CUptiResult CUPTIAPI cuptiProfilerDeviceSupported(CUpti_Profiler_DeviceSupported_Params *pParams);

/** @} */ /* END CUPTI_METRIC_API */
#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /*_CUPTI_PROFILER_TARGET_H_*/
