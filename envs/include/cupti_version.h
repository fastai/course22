/*
 * Copyright 2010-2018 NVIDIA Corporation.  All rights reserved.
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

#if !defined(_CUPTI_VERSION_H_)
#define _CUPTI_VERSION_H_

#include <cuda_stdint.h>
#include <cupti_result.h>

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_VERSION_API CUPTI Version
 * Function and macro to determine the CUPTI version.
 * @{
 */

/**
 * \brief The API version for this implementation of CUPTI.
 *
 * The API version for this implementation of CUPTI. This define along
 * with \ref cuptiGetVersion can be used to dynamically detect if the
 * version of CUPTI compiled against matches the version of the loaded
 * CUPTI library.
 *
 * v1 : CUDAToolsSDK 4.0
 * v2 : CUDAToolsSDK 4.1
 * v3 : CUDA Toolkit 5.0
 * v4 : CUDA Toolkit 5.5
 * v5 : CUDA Toolkit 6.0
 * v6 : CUDA Toolkit 6.5
 * v7 : CUDA Toolkit 6.5(with sm_52 support)
 * v8 : CUDA Toolkit 7.0
 * v9 : CUDA Toolkit 8.0
 * v10 : CUDA Toolkit 9.0
 * v11 : CUDA Toolkit 9.1
 * v12 : CUDA Toolkit 10.0, 10.1 and 10.2
 * v13 : CUDA Toolkit 11.0
 * v14 : CUDA Toolkit 11.1
 * v15 : CUDA Toolkit 11.2, 11.3 and 11.4
 * v16 : CUDA Toolkit 11.5
 * v17 : CUDA Toolkit 11.6
 * v18 : CUDA Toolkit 11.8
 * v19 : CUDA Toolkit 12.0
 */
#define CUPTI_API_VERSION 18

/**
 * \brief Get the CUPTI API version.
 *
 * Return the API version in \p *version.
 *
 * \param version Returns the version
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p version is NULL
 * \sa CUPTI_API_VERSION
 */
CUptiResult CUPTIAPI cuptiGetVersion(uint32_t *version);

/** @} */ /* END CUPTI_VERSION_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_VERSION_H_*/
