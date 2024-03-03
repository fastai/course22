/*
 * Copyright 2020-2022 NVIDIA Corporation.  All rights reserved.
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

#if !defined(_CUPTI_PCSAMPLING_H_)
#define _CUPTI_PCSAMPLING_H_

#include <cuda.h>
#include <stdint.h>
#include <stddef.h>
#include "cupti_result.h"

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#define ACTIVITY_RECORD_ALIGNMENT 8
#if defined(_WIN32) // Windows 32- and 64-bit
#define START_PACKED_ALIGNMENT __pragma(pack(push,1)) // exact fit - no padding
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_RECORD_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__) // GCC
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__ ((__packed__)) __attribute__ ((aligned (ACTIVITY_RECORD_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else // all other compilers
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_PCSAMPLING_API CUPTI PC Sampling API
 * Functions, types, and enums that implement the CUPTI PC Sampling API.
 * @{
 */

#ifndef CUPTI_PCSAMPLING_STRUCT_SIZE
#define CUPTI_PCSAMPLING_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

#ifndef CUPTI_STALL_REASON_STRING_SIZE
#define CUPTI_STALL_REASON_STRING_SIZE                                            128
#endif

/**
 * \brief PC Sampling collection mode
 */
typedef enum
{
  /**
   * INVALID Value
   */
  CUPTI_PC_SAMPLING_COLLECTION_MODE_INVALID                   = 0,
  /**
   * Continuous mode. Kernels are not serialized in this mode.
   */
  CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS                = 1,
  /**
   * Serialized mode. Kernels are serialized in this mode.
   */
  CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED         = 2,
} CUpti_PCSamplingCollectionMode;

/**
 * \brief PC Sampling stall reasons
 */
typedef struct PACKED_ALIGNMENT
{
  /**
   * [r] Collected stall reason index
   */
  uint32_t pcSamplingStallReasonIndex;
  /**
   * [r] Number of times the PC was sampled with the stallReason.
   */
  uint32_t samples;
} CUpti_PCSamplingStallReason;

/**
 * \brief PC Sampling data
 */
typedef struct PACKED_ALIGNMENT
{
  /**
   * [w] Size of the data structure.
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [r] Unique cubin id
   */
  uint64_t cubinCrc;
  /**
   * [r] PC offset
   */
  uint64_t pcOffset;
  /**
   * The function's unique symbol index in the module.
   */
  uint32_t functionIndex;
  /**
   * Padding
   */
  uint32_t pad;
  /**
   * [r] The function name. This name string might be shared across all the records
   * including records from activity APIs representing the same function, and so it should not be
   * modified or freed until post processing of all the records is done. Once done, it is user’s responsibility to
   * free the memory using free() function.
   */
  char* functionName;
  /**
   * [r] Collected stall reason count
   */
  size_t stallReasonCount;
  /**
   * [r] Stall reason id
   * Total samples
   */
  CUpti_PCSamplingStallReason *stallReason;
} CUpti_PCSamplingPCData;

/**
 * \brief PC Sampling output data format
 */
typedef enum
{
    CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_INVALID          = 0,
  /**
   * HW buffer data will be parsed during collection of data
   */
    CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED           = 1,
} CUpti_PCSamplingOutputDataFormat;

/**
 * \brief Collected PC Sampling data
 *
 */
typedef struct PACKED_ALIGNMENT
{
  /**
   * [w] Size of the data structure.
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Number of PCs to be collected
   */
  size_t collectNumPcs;
  /**
   * [r] Number of samples collected across all PCs.
   * It includes samples for user modules, samples for non-user kernels and dropped samples.
   * It includes counts for all non selected stall reasons.
   * CUPTI does not provide PC records for non-user kernels.
   * CUPTI does not provide PC records for instructions for which all selected stall reason metrics counts are zero.
   */
  uint64_t totalSamples;
  /**
   * [r] Number of samples that were dropped by hardware due to backpressure/overflow.
   */
  uint64_t droppedSamples;
  /**
   * [r] Number of PCs collected
   */
  size_t totalNumPcs;
  /**
   * [r] Number of PCs available for collection
   */
  size_t remainingNumPcs;
  /**
   * [r] Unique identifier for each range.
   * Data collected across multiple ranges in multiple buffers can be identified using range id.
   */
  uint64_t rangeId;
  /**
   * [r] Profiled PC data
   * This data struct should have enough memory to collect number of PCs mentioned in \brief collectNumPcs
   */
  CUpti_PCSamplingPCData *pPcData;
  /**
   * [r] Number of samples collected across all non user kernels PCs.
   * It includes samples for non-user kernels.
   * It includes counts for all non selected stall reasons as well.
   * CUPTI does not provide PC records for non-user kernels.
   */
  uint64_t nonUsrKernelsTotalSamples;

  /**
   * [r] Status of the hardware buffer.
   * CUPTI returns the error code CUPTI_ERROR_OUT_OF_MEMORY when hardware buffer is full.
   * When hardware buffer is full, user will get pc data as 0. To mitigate this issue, one or more of the below options can be tried:
   * 1. Increase the hardware buffer size using the attribute CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE
   * 2. Decrease the thread sleep span using the attribute CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN
   * 3. Decrease the sampling frequency using the attribute CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD
   */
  uint8_t hardwareBufferFull;
} CUpti_PCSamplingData;

/**
 * \brief PC Sampling configuration attributes
 *
 * PC Sampling configuration attribute types. These attributes can be read
 * using \ref cuptiPCSamplingGetConfigurationAttribute and can be written
 * using \ref cuptiPCSamplingSetConfigurationAttribute. Attributes marked
 * [r] can only be read using \ref cuptiPCSamplingGetConfigurationAttribute
 * [w] can only be written using \ref cuptiPCSamplingSetConfigurationAttribute
 * [rw] can be read using \ref cuptiPCSamplingGetConfigurationAttribute and
 * written using \ref cuptiPCSamplingSetConfigurationAttribute
 */
typedef enum
{
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_INVALID                            = 0,
  /**
   * [rw] Sampling period for PC Sampling.
   * DEFAULT - CUPTI defined value based on number of SMs
   * Valid values for the sampling
   * periods are between 5 to 31 both inclusive. This will set the
   * sampling period to (2^samplingPeriod) cycles.
   * For e.g. for sampling period = 5 to 31, cycles = 32, 64, 128,..., 2^31
   * Value is a uint32_t
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD                    = 1,
  /**
   * [w] Number of stall reasons to collect.
   * DEFAULT - All stall reasons will be collected
   * Value is a size_t
   * [w] Stall reasons to collect
   * DEFAULT - All stall reasons will be collected
   * Input value should be a pointer pointing to array of stall reason indexes
   * containing all the stall reason indexes to collect.
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON                       = 2,
  /**
   * [rw] Size of SW buffer for raw PC counter data downloaded from HW buffer
   * DEFAULT - 1 MB, which can accommodate approximately 5500 PCs
   * with all stall reasons
   * Approximately it takes 16 Bytes (and some fixed size memory)
   * to accommodate one PC with one stall reason
   * For e.g. 1 PC with 1 stall reason = 32 Bytes
   *          1 PC with 2 stall reason = 48 Bytes
   *          1 PC with 4 stall reason = 96 Bytes
   * Value is a size_t
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE                = 3,
  /**
   * [rw] Size of HW buffer in bytes
   * DEFAULT - 512 MB
   * If sampling period is too less, HW buffer can overflow
   * and drop PC data
   * Value is a size_t
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE               = 4,
  /**
   * [rw] PC Sampling collection mode
   * DEFAULT - CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS
   * Input value should be of type \ref CUpti_PCSamplingCollectionMode.
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE                    = 5,
  /**
   * [rw] Control over PC Sampling data collection range
   * Default - 0
   * 1 - Allows user to start and stop PC Sampling using APIs -
   * \ref cuptiPCSamplingStart() - Start PC Sampling
   * \ref cuptiPCSamplingStop() - Stop PC Sampling
   * Value is a uint32_t
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL          = 6,
  /**
   * [w] Value for output data format
   * Default - CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED
   * Input value should be of type \ref CUpti_PCSamplingOutputDataFormat.
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT                 = 7,
  /**
   * [w] Data buffer to hold collected PC Sampling data PARSED_DATA
   * Default - none.
   * Buffer type is void * which can point to PARSED_DATA
   * Refer \ref CUpti_PCSamplingData for buffer format for PARSED_DATA
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER               = 8,
  /**
   * [rw] Control sleep time of the worker threads created by CUPTI for various PC sampling operations.
   * CUPTI creates multiple worker threads to offload certain operations to these threads. This includes decoding of HW data to
   * the CUPTI PC sampling data and correlating PC data to SASS instructions. CUPTI wakes up these threads periodically.
   * Default - 100 milliseconds.
   * Value is a uint32_t
   */
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN  = 9,
  CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_FORCE_INT                          = 0x7fffffff,
} CUpti_PCSamplingConfigurationAttributeType;

/**
 * \brief PC sampling configuration information structure
 *
 * This structure provides \ref CUpti_PCSamplingConfigurationAttributeType which can be configured
 * or queried for PC sampling configuration
 */
typedef struct
{
  /**
   * Refer \ref CUpti_PCSamplingConfigurationAttributeType for all supported attribute types
   */
  CUpti_PCSamplingConfigurationAttributeType attributeType;
  /*
   * Configure or query status for \p attributeType
   * CUPTI_SUCCESS for valid \p attributeType and \p attributeData
   * CUPTI_ERROR_INVALID_OPERATION if \p attributeData is not valid
   * CUPTI_ERROR_INVALID_PARAMETER if \p attributeType is not valid
   */
  CUptiResult attributeStatus;
  union
  {
    /**
     * Invalid Value
     */
    struct
    {
      uint64_t data[3];
    } invalidData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD
     */
    struct
    {
      uint32_t samplingPeriod;
    } samplingPeriodData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON
     */
    struct
    {
      size_t stallReasonCount;
      uint32_t *pStallReasonIndex;
    } stallReasonData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE
     */
    struct
    {
      size_t scratchBufferSize;
    } scratchBufferSizeData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE
     */
    struct
    {
      size_t hardwareBufferSize;
    } hardwareBufferSizeData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE
     */
    struct
    {
      CUpti_PCSamplingCollectionMode collectionMode;
    } collectionModeData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL
     */
    struct
    {
      uint32_t enableStartStopControl;
    } enableStartStopControlData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT
     */
    struct
    {
      CUpti_PCSamplingOutputDataFormat outputDataFormat;
    } outputDataFormatData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER
     */
    struct
    {
      void *samplingDataBuffer;
    } samplingDataBufferData;
    /**
     * Refer \ref CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN
     */
    struct
    {
      uint32_t workerThreadPeriodicSleepSpan;
    } workerThreadPeriodicSleepSpanData;
    
  } attributeData;
} CUpti_PCSamplingConfigurationInfo;

/**
 * \brief PC sampling configuration structure
 *
 * This structure configures PC sampling using \ref cuptiPCSamplingSetConfigurationAttribute
 * and queries PC sampling default configuration using \ref cuptiPCSamplingGetConfigurationAttribute
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingConfigurationInfoParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
  /**
   * [w] Number of attributes to configure using \ref cuptiPCSamplingSetConfigurationAttribute or query
   * using \ref cuptiPCSamplingGetConfigurationAttribute
   */
  size_t numAttributes;
  /**
   * Refer \ref CUpti_PCSamplingConfigurationInfo
   */
  CUpti_PCSamplingConfigurationInfo *pPCSamplingConfigurationInfo;
} CUpti_PCSamplingConfigurationInfoParams;
#define CUpti_PCSamplingConfigurationInfoParamsSize                 CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingConfigurationInfoParams,pPCSamplingConfigurationInfo)

/**
 * \brief Write PC Sampling configuration attribute.
 *
 * \param pParams A pointer to \ref CUpti_PCSamplingConfigurationInfoParams
 * containing PC sampling configuration.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called with
 * some invalid \p attrib.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if attribute \p value is not valid
 * or any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingSetConfigurationAttribute(CUpti_PCSamplingConfigurationInfoParams *pParams);

/**
 * \brief Read PC Sampling configuration attribute.
 *
 * \param pParams A pointer to \ref CUpti_PCSamplingConfigurationInfoParams
 * containing PC sampling configuration.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called with
 * some invalid attribute.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p attrib is not valid
 * or any \p pParams is not valid
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT indicates that
 * the \p value buffer is too small to hold the attribute value
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingGetConfigurationAttribute(CUpti_PCSamplingConfigurationInfoParams *pParams);

/**
 * \brief Params for cuptiPCSamplingEnable
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingGetDataParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
  /**
   * \param pcSamplingData Data buffer to hold collected PC Sampling data PARSED_DATA
   * Buffer type is void * which can point to PARSED_DATA
   * Refer \ref CUpti_PCSamplingData for buffer format for PARSED_DATA
   */
  void *pcSamplingData;
} CUpti_PCSamplingGetDataParams;
#define CUpti_PCSamplingGetDataParamsSize                           CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetDataParams, pcSamplingData)
/**
 * \brief Flush GPU PC sampling data periodically.
 *
 * Flushing of GPU PC Sampling data is required at following point to maintain uniqueness of PCs:
 * For \brief CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS, after every module load-unload-load
 * For \brief CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED, after every kernel ends
 * If configuration option \brief CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL
 * is enabled, then after every range end i.e. \brief cuptiPCSamplingStop()
 *
 * If application is profiled in \brief CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS, with disabled
 * \brief CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL, and there is no module unload,
 * user can collect data in two ways:
 * Use \brief cuptiPCSamplingGetData() API periodically
 * Use \brief cuptiPCSamplingDisable() on application exit and read GPU PC sampling data from sampling
 * data buffer passed during configuration.
 * Note: In case, \brief cuptiPCSamplingGetData() API is not called periodically, then sampling data buffer
 * passed during configuration should be large enough to hold all PCs data.
 *       \brief cuptiPCSamplingGetData() API never does device synchronization.
 *       It is possible that when the API is called there is some unconsumed data from the HW buffer. In this case
 * CUPTI provides only the data available with it at that moment.
 *
 * \param Refer \ref CUpti_PCSamplingGetDataParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called without
 * enabling PC sampling.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * \retval CUPTI_ERROR_OUT_OF_MEMORY indicates that the HW buffer is full
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingGetData(CUpti_PCSamplingGetDataParams *pParams);

/**
 * \brief Params for cuptiPCSamplingEnable
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingEnableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
} CUpti_PCSamplingEnableParams;
#define CUpti_PCSamplingEnableParamsSize                           CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingEnableParams, ctx)

/**
 * \brief Enable PC sampling.
 *
 * \param Refer \ref CUpti_PCSamplingEnableParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingEnable(CUpti_PCSamplingEnableParams *pParams);

/**
 * \brief Params for cuptiPCSamplingDisable
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
} CUpti_PCSamplingDisableParams;
#define CUpti_PCSamplingDisableParamsSize                           CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingDisableParams, ctx)

/**
 * \brief Disable PC sampling.
 *
 * For application which doesn't destroy the CUDA context explicitly,
 * this API does the PC Sampling tear-down, joins threads and copies PC records in the buffer provided
 * during the PC sampling configuration. PC records which can't be accommodated in the buffer are discarded.
 *
 * \param Refer \ref CUpti_PCSamplingDisableParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingDisable(CUpti_PCSamplingDisableParams *pParams);

/**
 * \brief Params for cuptiPCSamplingStart
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingStartParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
} CUpti_PCSamplingStartParams;
#define CUpti_PCSamplingStartParamsSize                             CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingStartParams, ctx)

/**
 * \brief Start PC sampling.
 *
 * User can collect PC Sampling data for user-defined range specified by Start/Stop APIs.
 * This API can be used to mark starting of range. Set configuration option
 * \brief CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL to use this API.
 *
 * \param Refer \ref CUpti_PCSamplingStartParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called with
 * incorrect PC Sampling configuration.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingStart(CUpti_PCSamplingStartParams *pParams);

/**
 * \brief Params for cuptiPCSamplingStop
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingStopParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
} CUpti_PCSamplingStopParams;
#define CUpti_PCSamplingStopParamsSize                              CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingStopParams, ctx)

/**
 * \brief Stop PC sampling.
 *
 * User can collect PC Sampling data for user-defined range specified by Start/Stop APIs.
 * This API can be used to mark end of range. Set configuration option
 * \brief CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL to use this API.
 *
 * \param Refer \ref CUpti_PCSamplingStopParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called with
 * incorrect PC Sampling configuration.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingStop(CUpti_PCSamplingStopParams *pParams);

/**
 * \brief Params for cuptiPCSamplingGetNumStallReasons
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingGetNumStallReasonsParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
  /**
   * [r] Number of stall reasons
   */
  size_t *numStallReasons;
} CUpti_PCSamplingGetNumStallReasonsParams;
#define CUpti_PCSamplingGetNumStallReasonsParamsSize                CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetNumStallReasonsParams, numStallReasons)

/**
 * \brief Get PC sampling stall reason count.
 *
 * \param Refer \ref CUpti_PCSamplingGetNumStallReasonsParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingGetNumStallReasons(CUpti_PCSamplingGetNumStallReasonsParams *pParams);

/**
 * \brief Params for cuptiPCSamplingGetStallReasons
 */
typedef struct
{
  /**
   * [w] Size of the data structure i.e. CUpti_PCSamplingGetStallReasonsParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Assign to NULL
   */
  void* pPriv;
  /**
   * [w] CUcontext
   */
  CUcontext ctx;
  /**
   * [w] Number of stall reasons
   */
  size_t numStallReasons;
  /**
   * [r] Stall reason index
   */
  uint32_t *stallReasonIndex;
  /**
   * [r] Stall reasons name
   */
  char **stallReasons;
} CUpti_PCSamplingGetStallReasonsParams;
#define CUpti_PCSamplingGetStallReasonsParamsSize                   CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetStallReasonsParams, stallReasons)

/**
 * \brief Get PC sampling stall reasons.
 *
 * \param Refer \ref CUpti_PCSamplingGetStallReasonsParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device
 * does not support the API
 */
CUptiResult CUPTIAPI cuptiPCSamplingGetStallReasons(CUpti_PCSamplingGetStallReasonsParams *pParams);

/**
 * \brief Params for cuptiGetSassToSourceCorrelation
 */
typedef struct {
  /**
   * [w] Size of the data structure i.e. CUpti_GetSassToSourceCorrelationParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Pointer to cubin binary where function belongs.
   */
  const void* cubin;
  /**
   * [w] Function name to which PC belongs.
   */
  const char *functionName;
  /**
   * [w] Size of cubin binary.
   */
  size_t cubinSize;
  /**
   * [r] Line number in the source code.
   */
  uint32_t lineNumber;
  /**
   * [w] PC offset
   */
  uint64_t pcOffset;
  /**
   * [r] Path for the source file.
   */
  char *fileName;
  /**
   * [r] Path for the directory of source file.
   */
  char *dirName;
} CUpti_GetSassToSourceCorrelationParams;
#define CUpti_GetSassToSourceCorrelationParamsSize     CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_GetSassToSourceCorrelationParams, dirName)

/**
 * \brief SASS to Source correlation.
 *
 * \param Refer \ref CUpti_GetSassToSourceCorrelationParams
 *
 * It is expected from user to free allocated memory for fileName and dirName after use.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if either of the parameters cubin or functionName
 * is NULL or cubinSize is zero or size field is not set correctly.
 * \retval CUPTI_ERROR_INVALID_MODULE provided cubin is invalid.
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred.
 * This error code is also used for cases when the function is not present in the module.
 * A better error code will be returned in the future release.
 */
CUptiResult CUPTIAPI cuptiGetSassToSourceCorrelation(CUpti_GetSassToSourceCorrelationParams *pParams);

/**
 * \brief Params for cuptiGetCubinCrc
 */
typedef struct {
  /**
   * [w] Size of configuration structure.
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * [w] Size of cubin binary.
   */
  size_t cubinSize;
  /**
   * [w] Pointer to cubin binary
   */
  const void* cubin;
  /**
   * [r] Computed CRC will be stored in it.
   */
  uint64_t cubinCrc;
} CUpti_GetCubinCrcParams;
#define CUpti_GetCubinCrcParamsSize     CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_GetCubinCrcParams, cubinCrc)

/**
 * \brief Get the CRC of cubin.
 *
 * This function returns the CRC of provided cubin binary.
 *
 * \param Refer \ref CUpti_GetCubinCrcParams
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if parameter cubin is NULL or
 * provided cubinSize is zero or size field is not set.
 */
CUptiResult CUPTIAPI cuptiGetCubinCrc(CUpti_GetCubinCrcParams *pParams);

/**
 * \brief Function type for callback used by CUPTI to request crc of
 * loaded module.
 *
 * This callback function ask for crc of provided module in function.
 * The provided crc will be stored in PC sampling records i.e. in the field 'cubinCrc' of the PC sampling
 * struct CUpti_PCSamplingPCData. The CRC is uses during the offline source correlation to uniquely identify the module.
 *
 * \param cubin The pointer to cubin binary
 * \param cubinSize The size of cubin binary.
 * \param cubinCrc Returns the computed crc of cubin.
 */
typedef void (CUPTIAPI *CUpti_ComputeCrcCallbackFunc)(
    const void* cubin,
    size_t cubinSize,
    uint64_t *cubinCrc);

/**
 * \brief Register callback function with CUPTI to use
 * your own algorithm to compute cubin crc.
 *
 * This function registers a callback function and it gets called
 * from CUPTI when a CUDA module is loaded.
 *
 * \param funcComputeCubinCrc callback is invoked when a CUDA module
 * is loaded.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p funcComputeCubinCrc is NULL.
 */
CUptiResult CUPTIAPI cuptiRegisterComputeCrcCallback(CUpti_ComputeCrcCallbackFunc funcComputeCubinCrc);

/** @} */ /* END CUPTI_PCSAMPLING_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_PCSAMPLING_H_*/
