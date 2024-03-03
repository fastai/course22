/*
 * Copyright 2015-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample OpenACC app computing a saxpy kernel.
 * Data collection of OpenACC records via CUPTI is implemented
 * in a shared library attached at runtime.
 */

// System headers
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// OpenAcc headers
#include <openacc.h>

// OpenACC kernels
static void
OpenaccKernel(
    const int N,
    const float A,
    float *pX,
    float *pY)
{
    int i;
#pragma acc kernels
    for (i = 0; i < N; ++i)
    {
        pY[i] = A * pX[i];
    }
}

// Functions
static void
SetClear(
    const int N,
    float *pA)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        pA[i] = 0.0;
    }
}

static void
InitializeVector(
    const int N,
    const float Mult,
    float *pX)
{
    int i;

    // CUPTI OpenACC only supports NVIDIA devices
#pragma acc kernels

#if (!defined(HOST_ARCH_PPC))
    assert(acc_on_device(acc_device_nvidia));
#endif

    for (i = 0; i < N; ++i)
    {
        pX[i] = Mult * i;
    }
}

int
main(
    int argc,
    char *argv[])
{
    int N = 32000;

    float *pX = new float[N];
    float *pY = new float[N];

    // initialize data
    InitializeVector(N, 0.5, pX);
    SetClear(N, pY);

    // run saxpy kernel
    OpenaccKernel(N, 2.0f, pX, pY);

    // cleanup
    delete[] pX;
    delete[] pY;

    exit(EXIT_SUCCESS);
}
