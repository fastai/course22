/*
 * Copyright 2010-2017 NVIDIA Corporation.  All rights reserved.
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

#if !defined(_CUPTI_H_)
#define _CUPTI_H_

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef NOMINMAX
#include <windows.h>
#else
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif
#endif

#include <cuda.h>
#include <cupti_result.h>
#include <cupti_version.h>

/* Activity, callback, event and metric APIs */
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_events.h>
#include <cupti_metrics.h>

/* Runtime, driver, and nvtx function identifiers */
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#include <cupti_nvtx_cbid.h>

/* To support function parameter structures for obsoleted API. See
   cuda.h for the actual definition of these structures. */
typedef unsigned int CUdeviceptr_v1;
typedef struct CUDA_MEMCPY2D_v1_st { int dummy; } CUDA_MEMCPY2D_v1;
typedef struct CUDA_MEMCPY3D_v1_st { int dummy; } CUDA_MEMCPY3D_v1;
typedef struct CUDA_ARRAY_DESCRIPTOR_v1_st { int dummy; } CUDA_ARRAY_DESCRIPTOR_v1;
typedef struct CUDA_ARRAY3D_DESCRIPTOR_v1_st { int dummy; } CUDA_ARRAY3D_DESCRIPTOR_v1;

/* Function parameter structures */
#include <generated_cuda_runtime_api_meta.h>
#include <generated_cuda_meta.h>

/* The following parameter structures cannot be included unless a
   header that defines GL_VERSION is included before including them.
   If these are needed then make sure such a header is included
   already. */
#ifdef GL_VERSION
#include <generated_cuda_gl_interop_meta.h>
#include <generated_cudaGL_meta.h>
#endif

//#include <generated_nvtx_meta.h>

/* The following parameter structures cannot be included by default as
   they are not guaranteed to be available on all systems. Uncomment
   the includes that are available, or use the include explicitly. */
#if defined(__linux__)
//#include <generated_cuda_vdpau_interop_meta.h>
//#include <generated_cudaVDPAU_meta.h>
#endif

#ifdef _WIN32
//#include <generated_cuda_d3d9_interop_meta.h>
//#include <generated_cuda_d3d10_interop_meta.h>
//#include <generated_cuda_d3d11_interop_meta.h>
//#include <generated_cudaD3D9_meta.h>
//#include <generated_cudaD3D10_meta.h>
//#include <generated_cudaD3D11_meta.h>
#endif

#endif /*_CUPTI_H_*/


