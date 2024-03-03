#if !defined(_CUPTI_TARGET_H_)
#define _CUPTI_TARGET_H_

/*
CUPTI profiler target API's
This file contains the CUPTI profiling API's.
*/
#include <cupti_result.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

#ifndef CUPTI_PROFILER_STRUCT_SIZE
#define CUPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

typedef struct CUpti_Device_GetChipName_Params
{
    size_t structSize;                                      //!< [in]
    void* pPriv;                                            //!< [in] assign to NULL

    size_t deviceIndex;                                     //!< [in]
    const char* pChipName;                                  //!< [out]
} CUpti_Device_GetChipName_Params;

#define CUpti_Device_GetChipName_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Device_GetChipName_Params, pChipName)
CUptiResult CUPTIAPI cuptiDeviceGetChipName(CUpti_Device_GetChipName_Params *pParams);

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif
