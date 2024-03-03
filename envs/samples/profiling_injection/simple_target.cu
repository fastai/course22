// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// A simple sample CUDA program to test injecting CUPTI
// Profiler API calls on.  8 launches should be detected with
// an increasing amount of work per call.

#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "driver_types.h"
#include "stdio.h"

#include <iostream>
using ::std::cout;
using ::std::endl;

void __global__ sum_red(double * out, double * in, size_t n)
{
    double sum = 0.0;

    for (size_t idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        sum += in[idx];
    }

    out[threadIdx.x] = sum;
}

int main()
{
    int n = 10000000;
    int thds = 256;

    // Point for injection library to intercept
    //
    // Could also intercept main() or other functions that will
    // be guaranteed to run before the first cuda call that should
    // be modeled, in which case this isn't needed for a Runtime API
    // example code.
    //cuInit(0);

    double * h_in;
    h_in = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        h_in[i] = rand() / (double)RAND_MAX;
    }
    double expect = 0.0;
    for (int i = 0; i < n; i++)
    {
        expect += h_in[i];
    }
    cout << "Expect reduction to equal " << expect << endl;

    double * h_out;
    h_out = (double *)malloc(thds * sizeof(double));

    double * d_in;
    cudaMalloc(&d_in, n * sizeof(double));
    cudaMemcpy(d_in, h_in, n * sizeof(double), cudaMemcpyHostToDevice);

    double * d_out;
    cudaMalloc(&d_out, thds * sizeof(double));

    sum_red<<<1, thds>>>(d_out, d_in, n/8);
    sum_red<<<1, thds>>>(d_out, d_in, n/7);
    sum_red<<<1, thds>>>(d_out, d_in, n/6);
    sum_red<<<1, thds>>>(d_out, d_in, n/5);
    sum_red<<<1, thds>>>(d_out, d_in, n/4);
    sum_red<<<1, thds>>>(d_out, d_in, n/3);
    sum_red<<<1, thds>>>(d_out, d_in, n/2);
    sum_red<<<1, thds>>>(d_out, d_in, n/1);

    cudaMemcpy(h_out, d_out, thds * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < thds; i++)
    {
        sum += h_out[i];
    }

    cout << "Sum reduction = " << sum << endl;
}
