/*
 * Copyright 2013-2017 NVIDIA Corporation.  All rights reserved.
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

typedef enum {
  CUPTI_CBID_NVTX_INVALID                               = 0,
  CUPTI_CBID_NVTX_nvtxMarkA                             = 1,
  CUPTI_CBID_NVTX_nvtxMarkW                             = 2,
  CUPTI_CBID_NVTX_nvtxMarkEx                            = 3,
  CUPTI_CBID_NVTX_nvtxRangeStartA                       = 4,
  CUPTI_CBID_NVTX_nvtxRangeStartW                       = 5,
  CUPTI_CBID_NVTX_nvtxRangeStartEx                      = 6,
  CUPTI_CBID_NVTX_nvtxRangeEnd                          = 7,
  CUPTI_CBID_NVTX_nvtxRangePushA                        = 8,
  CUPTI_CBID_NVTX_nvtxRangePushW                        = 9,
  CUPTI_CBID_NVTX_nvtxRangePushEx                       = 10,
  CUPTI_CBID_NVTX_nvtxRangePop                          = 11,
  CUPTI_CBID_NVTX_nvtxNameCategoryA                     = 12,
  CUPTI_CBID_NVTX_nvtxNameCategoryW                     = 13,
  CUPTI_CBID_NVTX_nvtxNameOsThreadA                     = 14,
  CUPTI_CBID_NVTX_nvtxNameOsThreadW                     = 15,
  CUPTI_CBID_NVTX_nvtxNameCuDeviceA                     = 16,
  CUPTI_CBID_NVTX_nvtxNameCuDeviceW                     = 17,
  CUPTI_CBID_NVTX_nvtxNameCuContextA                    = 18,
  CUPTI_CBID_NVTX_nvtxNameCuContextW                    = 19,
  CUPTI_CBID_NVTX_nvtxNameCuStreamA                     = 20,
  CUPTI_CBID_NVTX_nvtxNameCuStreamW                     = 21,
  CUPTI_CBID_NVTX_nvtxNameCuEventA                      = 22,
  CUPTI_CBID_NVTX_nvtxNameCuEventW                      = 23,
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceA                   = 24,
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceW                   = 25,
  CUPTI_CBID_NVTX_nvtxNameCudaStreamA                   = 26,
  CUPTI_CBID_NVTX_nvtxNameCudaStreamW                   = 27,
  CUPTI_CBID_NVTX_nvtxNameCudaEventA                    = 28,
  CUPTI_CBID_NVTX_nvtxNameCudaEventW                    = 29,
  CUPTI_CBID_NVTX_nvtxDomainMarkEx                      = 30,
  CUPTI_CBID_NVTX_nvtxDomainRangeStartEx                = 31,
  CUPTI_CBID_NVTX_nvtxDomainRangeEnd                    = 32,
  CUPTI_CBID_NVTX_nvtxDomainRangePushEx                 = 33,
  CUPTI_CBID_NVTX_nvtxDomainRangePop                    = 34,
  CUPTI_CBID_NVTX_nvtxDomainResourceCreate              = 35,
  CUPTI_CBID_NVTX_nvtxDomainResourceDestroy             = 36,
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryA               = 37,
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryW               = 38,
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringA             = 39,
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringW             = 40,
  CUPTI_CBID_NVTX_nvtxDomainCreateA                     = 41,
  CUPTI_CBID_NVTX_nvtxDomainCreateW                     = 42,
  CUPTI_CBID_NVTX_nvtxDomainDestroy                     = 43,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserCreate              = 44,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserDestroy             = 45,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireStart        = 46,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireFailed       = 47,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireSuccess      = 48,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserReleasing           = 49,
  CUPTI_CBID_NVTX_SIZE,
  CUPTI_CBID_NVTX_FORCE_INT                             = 0x7fffffff
} CUpti_nvtx_api_trace_cbid;

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif    
