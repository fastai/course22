#if !defined(_CUPTI_PCSAMPLING_UTIL_H_)
#define _CUPTI_PCSAMPLING_UTIL_H_

#include <cupti_pcsampling.h>
#include <fstream>

#ifndef CUPTIUTILAPI
#ifdef _WIN32
#define CUPTIUTILAPI __stdcall
#else
#define CUPTIUTILAPI
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

#ifndef CUPTI_UTIL_STRUCT_SIZE
#define CUPTI_UTIL_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

#ifndef CHECK_PC_SAMPLING_STRUCT_FIELD_EXISTS
#define CHECK_PC_SAMPLING_STRUCT_FIELD_EXISTS(type, member, structSize)    \
    (offsetof(type, member) < structSize)
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__)
    #pragma GCC visibility push(default)
#endif

namespace CUPTI { namespace PcSamplingUtil {

/**
 * \defgroup CUPTI_PCSAMPLING_UTILITY CUPTI PC Sampling Utility API
 * Functions, types, and enums that implement the CUPTI PC Sampling Utility API.
 * @{
 */

/**
 * \brief Header info will be stored in file.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Version of file format.
   */
  uint32_t version;
  /**
   * Total number of buffers present in the file.
   */
  uint32_t totalBuffers;
} Header;

/**
 * \brief BufferInfo will be stored in the file for every buffer
 *  i.e for every call of UtilDumpPcSamplingBufferInFile() API.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Total number of PC records.
   */
  uint64_t recordCount;
  /**
   * Count of all stall reasons supported on the GPU
   */
  size_t numStallReasons;
  /**
   * Total number of stall reasons in single record.
   */
  uint64_t numSelectedStallReasons;
  /**
   * Buffer size in Bytes.
   */
  uint64_t bufferByteSize;
} BufferInfo;

/**
 * \brief All available stall reasons name and respective indexes
 * will be stored in it.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Number of all available stall reasons
   */
  size_t numStallReasons;
  /**
   * Stall reasons names of all available stall reasons
   */
  char **stallReasons;
  /**
   * Stall reason index of all available stall reasons
   */
  uint32_t *stallReasonIndex;
} PcSamplingStallReasons;

typedef enum {
  /**
   * Invalid buffer type.
   */
  PC_SAMPLING_BUFFER_INVALID             = 0,
  /**
   * Refers to CUpti_PCSamplingData buffer.
   */
  PC_SAMPLING_BUFFER_PC_TO_COUNTER_DATA  = 1
} PcSamplingBufferType;

/**
 * \brief CUPTI PC sampling utility API result codes.
 *
 * Error and result codes returned by CUPTI PC sampling utility API.
 */
typedef enum {
  /**
   * No error
   */
  CUPTI_UTIL_SUCCESS                                       = 0,
  /**
   * One or more of the parameters are invalid.
   */
  CUPTI_UTIL_ERROR_INVALID_PARAMETER                       = 1,
  /**
   * Unable to create a new file
   */
  CUPTI_UTIL_ERROR_UNABLE_TO_CREATE_FILE                   = 2,
  /**
   * Unable to open a file
   */
  CUPTI_UTIL_ERROR_UNABLE_TO_OPEN_FILE                     = 3,
  /**
   * Read or write operation failed
   */
  CUPTI_UTIL_ERROR_READ_WRITE_OPERATION_FAILED             = 4,
  /**
   * Provided file handle is corrupted.
   */
  CUPTI_UTIL_ERROR_FILE_HANDLE_CORRUPTED                   = 5,
  /**
   * seek operation failed.
   */
  CUPTI_UTIL_ERROR_SEEK_OPERATION_FAILED                   = 6,
  /**
   * Unable to allocate enough memory to perform the requested
   * operation.
   */
  CUPTI_UTIL_ERROR_OUT_OF_MEMORY                           = 7,
  /**
   * An unknown internal error has occurred.
   */
  CUPTI_UTIL_ERROR_UNKNOWN                                 = 999,
  CUPTI_UTIL_ERROR_FORCE_INT                               = 0x7fffffff
} CUptiUtilResult;

/**
 * \brief Params for \ref CuptiUtilPutPcSampData
 */
typedef struct {
  /**
   * Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * Type of buffer to store in file
   */
  PcSamplingBufferType bufferType;
  /**
   * PC sampling buffer.
   */
  void *pSamplingData;
  /**
   * Number of configured attributes
   */
  size_t numAttributes;
  /**
   * Refer \ref CUpti_PCSamplingConfigurationInfo
   * It is expected to provide configuration details of at least
   * CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON attribute.
   */
  CUpti_PCSamplingConfigurationInfo *pPCSamplingConfigurationInfo;
  /**
   * Refer \ref PcSamplingStallReasons.
   */
  PcSamplingStallReasons *pPcSamplingStallReasons;
  /**
   * File name to store buffer into it.
   */
  const char* fileName;
} CUptiUtil_PutPcSampDataParams;
#define CUptiUtil_PutPcSampDataParamsSize                   CUPTI_UTIL_STRUCT_SIZE(CUptiUtil_PutPcSampDataParams, fileName)

/**
 * \brief Dump PC sampling data into the file.
 *
 * This API can be called multiple times.
 * It will append buffer in the file.
 * For every buffer it will store BufferInfo
 * so that before retrieving data it will help to allocate buffer
 * to store retrieved data.
 * This API creates file if file does not present.
 * If stallReasonIndex or stallReasons pointer of \ref CUptiUtil_PutPcSampDataParams is NULL
 * then stall reasons data  will not be stored in file.
 * It is expected to store all available stall reason data at least once to refer it during
 * offline correlation.
 *
 * \retval CUPTI_UTIL_SUCCESS
 * \retval CUPTI_UTIL_ERROR_INVALID_PARAMETER error out if buffer type is invalid
 * or if either of pSamplingData, pParams pointer is NULL or stall reason configuration details not provided
 * or filename is empty.
 * \retval CUPTI_UTIL_ERROR_UNABLE_TO_CREATE_FILE
 * \retval CUPTI_UTIL_ERROR_UNABLE_TO_OPEN_FILE
 * \retval CUPTI_UTIL_ERROR_READ_WRITE_OPERATION_FAILED
 */
CUptiUtilResult CUPTIUTILAPI CuptiUtilPutPcSampData(CUptiUtil_PutPcSampDataParams *pParams);

/**
 * \brief Params for \ref CuptiUtilGetHeaderData
 */
typedef struct {
  /**
   * Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * File handle.
   */
  std::ifstream *fileHandler;
  /**
   * Header Info.
   */
  Header headerInfo;

} CUptiUtil_GetHeaderDataParams;
#define CUptiUtil_GetHeaderDataParamsSize                   CUPTI_UTIL_STRUCT_SIZE(CUptiUtil_GetHeaderDataParams, headerInfo)

/**
 * \brief Get header data of file.
 *
 * This API must be called once initially while retrieving data from file.
 * \ref Header structure, it gives info about total number
 * of buffers present in the file.
 *
 * \retval CUPTI_UTIL_SUCCESS
 * \retval CUPTI_UTIL_ERROR_INVALID_PARAMETER error out if either of pParam or fileHandle is NULL or param struct size is incorrect.
 * \retval CUPTI_UTIL_ERROR_FILE_HANDLE_CORRUPTED file handle is not in good state to read data from file
 * \retval CUPTI_UTIL_ERROR_READ_WRITE_OPERATION_FAILED  failed to read data from file.
 */
CUptiUtilResult CUPTIUTILAPI CuptiUtilGetHeaderData(CUptiUtil_GetHeaderDataParams *pParams);

/**
 * \brief Params for \ref CuptiUtilGetBufferInfo
 */
typedef struct {
  /**
   * Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * File handle.
   */
  std::ifstream *fileHandler;
  /**
   * Buffer Info.
   */
  BufferInfo bufferInfoData;
} CUptiUtil_GetBufferInfoParams;
#define CUptiUtil_GetBufferInfoParamsSize                   CUPTI_UTIL_STRUCT_SIZE(CUptiUtil_GetBufferInfoParams, bufferInfoData)

/**
 * \brief Get buffer info data of file.
 *
 * This API must be called every time before calling CuptiUtilGetPcSampData API.
 * \ref BufferInfo structure, it gives info about recordCount and stallReasonCount
 * of every record in the buffer. This will help to allocate exact buffer to retrieve data into it.
 *
 * \retval CUPTI_UTIL_SUCCESS
 * \retval CUPTI_UTIL_ERROR_INVALID_PARAMETER error out if either of pParam or fileHandle is NULL or param struct size is incorrect.
 * \retval CUPTI_UTIL_ERROR_FILE_HANDLE_CORRUPTED file handle is not in good state to read data from file.
 * \retval CUPTI_UTIL_ERROR_READ_WRITE_OPERATION_FAILED failed to read data from file.
 */
CUptiUtilResult CUPTIUTILAPI CuptiUtilGetBufferInfo(CUptiUtil_GetBufferInfoParams *pParams);

/**
 * \brief Params for \ref CuptiUtilGetPcSampData
 */
typedef struct {
  /**
   * Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * File handle.
   */
  std::ifstream *fileHandler;
  /**
   * Type of buffer to store in file
   */
  PcSamplingBufferType bufferType;
  /**
   * Pointer to collected buffer info using \ref CuptiUtilGetBufferInfo
   */
  BufferInfo *pBufferInfoData;
  /**
   * Pointer to allocated memory to store retrieved data from file.
   */
  void *pSamplingData;
  /**
   * Number of configuration attributes
   */
  size_t numAttributes;
  /**
   * Refer \ref CUpti_PCSamplingConfigurationInfo
   */
  CUpti_PCSamplingConfigurationInfo *pPCSamplingConfigurationInfo;
  /**
   * Refer \ref PcSamplingStallReasons.
   * For stallReasons field of \ref PcSamplingStallReasons it is expected to
   * allocate memory for each string element of array.
   */
  PcSamplingStallReasons *pPcSamplingStallReasons;
} CUptiUtil_GetPcSampDataParams;
#define CUptiUtil_GetPcSampDataParamsSize                   CUPTI_UTIL_STRUCT_SIZE(CUptiUtil_GetPcSampDataParams, pPcSamplingStallReasons)

/**
 * \brief Retrieve PC sampling data from file into allocated buffer.
 *
 * This API must be called after CuptiUtilGetBufferInfo API.
 * It will retrieve data from file into allocated buffer.
 *
 * \retval CUPTI_UTIL_SUCCESS
 * \retval CUPTI_UTIL_ERROR_INVALID_PARAMETER error out if buffer type is invalid
 * or if either of pSampData, pParams is NULL. If pPcSamplingStallReasons is not NULL then
 * error out if either of stallReasonIndex, stallReasons or stallReasons array element pointer is NULL.
 * or filename is empty.
 * \retval CUPTI_UTIL_ERROR_READ_WRITE_OPERATION_FAILED
 * \retval CUPTI_UTIL_ERROR_FILE_HANDLE_CORRUPTED file handle is not in good state to read data from file.
 */
CUptiUtilResult CUPTIUTILAPI CuptiUtilGetPcSampData(CUptiUtil_GetPcSampDataParams *pParams);

/**
 * \brief Params for \ref CuptiUtilMergePcSampData
 */
typedef struct
{
  /**
   * Size of the data structure i.e. CUpti_PCSamplingDisableParamsSize
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  size_t size;
  /**
   * Number of buffers to merge.
   */
  size_t numberOfBuffers;
  /**
   * Pointer to array of buffers to merge
   */
  CUpti_PCSamplingData *PcSampDataBuffer;
  /**
   * Pointer to array of merged buffers as per the range id.
   */
  CUpti_PCSamplingData **MergedPcSampDataBuffers;
  /**
   * Number of merged buffers.
   */
  size_t *numMergedBuffer;
} CUptiUtil_MergePcSampDataParams;
#define CUptiUtil_MergePcSampDataParamsSize                   CUPTI_UTIL_STRUCT_SIZE(CUptiUtil_MergePcSampDataParams, numMergedBuffer)

/**
 * \brief Merge PC sampling data range id wise.
 *
 * This API merge PC sampling data range id wise.
 * It allocates memory for merged data and fill data in it
 * and provide buffer pointer in MergedPcSampDataBuffers field.
 * It is expected from user to free merge data buffers after use.
 *
 * \retval CUPTI_UTIL_SUCCESS
 * \retval CUPTI_UTIL_ERROR_INVALID_PARAMETER error out if param struct size is invalid
 * or count of buffers to merge is invalid i.e less than 1
 * or either of PcSampDataBuffer, MergedPcSampDataBuffers, numMergedBuffer is NULL
 * \retval CUPTI_UTIL_ERROR_OUT_OF_MEMORY Unable to allocate memory for merged buffer.
 */
CUptiUtilResult CUPTIUTILAPI CuptiUtilMergePcSampData(CUptiUtil_MergePcSampDataParams *pParams);

/** @} */ /* END CUPTI_PCSAMPLING_UTILITY */

} }

#if defined(__GNUC__)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif
