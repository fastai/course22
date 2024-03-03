/*
 * Copyright 2013-2018 NVIDIA Corporation.  All rights reserved.
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

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct nvtxMarkEx_params_st {
  const nvtxEventAttributes_t* eventAttrib;
} nvtxMarkEx_params;

typedef struct nvtxMarkA_params_st {
  const char* message;
} nvtxMarkA_params;

typedef struct nvtxMarkW_params_st {
  const wchar_t* message;
} nvtxMarkW_params;

typedef struct nvtxRangeStartEx_params_st {
  const nvtxEventAttributes_t* eventAttrib;
} nvtxRangeStartEx_params;

typedef struct nvtxRangeStartA_params_st {
  const char* message;
} nvtxRangeStartA_params;

typedef struct nvtxRangeStartW_params_st {
  const wchar_t* message;
} nvtxRangeStartW_params;

typedef struct nvtxRangeEnd_params_st {
  nvtxRangeId_t id;
} nvtxRangeEnd_params;

typedef struct nvtxRangePushEx_params_st {
  const nvtxEventAttributes_t* eventAttrib;
} nvtxRangePushEx_params;

typedef struct nvtxRangePushA_params_st {
  const char* message;
} nvtxRangePushA_params;

typedef struct nvtxRangePushW_params_st {
  const wchar_t* message;
} nvtxRangePushW_params;

typedef struct nvtxRangePop_params_st {
  /* WAR: Windows compiler doesn't allow empty structs */
  /* This field shouldn't be used */
  void *dummy;
} nvtxRangePop_params;

typedef struct nvtxNameCategoryA_params_st {
  uint32_t category;
  const char* name;
} nvtxNameCategoryA_params;

typedef struct nvtxNameCategoryW_params_st {
  uint32_t category;
  const wchar_t* name;
} nvtxNameCategoryW_params;

typedef struct nvtxNameOsThreadA_params_st {
  uint32_t threadId;
  const char* name;
} nvtxNameOsThreadA_params;

typedef struct nvtxNameOsThreadW_params_st {
  uint32_t threadId;
  const wchar_t* name;
} nvtxNameOsThreadW_params;

typedef struct nvtxNameCuDeviceA_params_st {
  CUdevice device;
  const char* name;
} nvtxNameCuDeviceA_params;

typedef struct nvtxNameCuDeviceW_params_st {
  CUdevice device;
  const wchar_t* name;
} nvtxNameCuDeviceW_params;

typedef struct nvtxNameCuContextA_params_st {
  CUcontext context;
  const char* name;
} nvtxNameCuContextA_params;

typedef struct nvtxNameCuContextW_params_st {
  CUcontext context;
  const wchar_t* name;
} nvtxNameCuContextW_params;

typedef struct nvtxNameCuStreamA_params_st {
  CUstream stream;
  const char* name;
} nvtxNameCuStreamA_params;

typedef struct nvtxNameCuStreamW_params_st {
  CUstream stream;
  const wchar_t* name;
} nvtxNameCuStreamW_params;

typedef struct nvtxNameCuEventA_params_st {
  CUevent event;
  const char* name;
} nvtxNameCuEventA_params;

typedef struct nvtxNameCuEventW_params_st {
  CUevent event;
  const wchar_t* name;
} nvtxNameCuEventW_params;

typedef struct nvtxNameCudaDeviceA_params_st {
  int device;
  const char* name;
} nvtxNameCudaDeviceA_params;

typedef struct nvtxNameCudaDeviceW_params_st {
  int device;
  const wchar_t* name;
} nvtxNameCudaDeviceW_params;

typedef struct nvtxNameCudaStreamA_params_st {
  cudaStream_t stream;
  const char* name;
} nvtxNameCudaStreamA_params;

typedef struct nvtxNameCudaStreamW_params_st {
  cudaStream_t stream;
  const wchar_t* name;
} nvtxNameCudaStreamW_params;

typedef struct nvtxNameCudaEventA_params_st {
  cudaEvent_t event;
  const char* name;
} nvtxNameCudaEventA_params;

typedef struct nvtxNameCudaEventW_params_st {
  cudaEvent_t event;
  const wchar_t* name;
} nvtxNameCudaEventW_params;

typedef struct nvtxDomainCreateA_params_st {
  const char* name;
} nvtxDomainCreateA_params;

typedef struct nvtxDomainDestroy_params_st {
  nvtxDomainHandle_t domain;
} nvtxDomainDestroy_params;

typedef struct nvtxDomainMarkEx_params_st {
  nvtxDomainHandle_t domain;
  nvtxMarkEx_params core;
} nvtxDomainMarkEx_params;

typedef struct nvtxDomainRangeStartEx_params_st {
  nvtxDomainHandle_t domain;
  nvtxRangeStartEx_params core;
} nvtxDomainRangeStartEx_params;

typedef struct nvtxDomainRangeEnd_params_st {
  nvtxDomainHandle_t domain;
  nvtxRangeEnd_params core;
} nvtxDomainRangeEnd_params;

typedef struct nvtxDomainRangePushEx_params_st {
  nvtxDomainHandle_t domain;
  nvtxRangePushEx_params core;
} nvtxDomainRangePushEx_params;

typedef struct nvtxDomainRangePop_params_st {
  nvtxDomainHandle_t domain;
} nvtxDomainRangePop_params;

typedef struct nvtxSyncUserCreate_params_st {
  nvtxDomainHandle_t domain;
  const nvtxSyncUserAttributes_t* attribs;
} nvtxSyncUserCreate_params;

typedef struct nvtxSyncUserCommon_params_st {
  nvtxSyncUser_t handle;
} nvtxSyncUserCommon_params;

typedef struct nvtxDomainRegisterStringA_params_st {
    nvtxDomainHandle_t domain;
    const char* string;
} nvtxDomainRegisterStringA_params;

typedef struct nvtxDomainRegisterStringW_params_st {
    nvtxDomainHandle_t domain;
    const char* string;
} nvtxDomainRegisterStringW_params;

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif
