// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates a very simple use case of the Checkpoint API -
// An array is saved to device, a checkpoint is saved capturing these initial
// values, the device memory is update with a new value, then restored to
// initial value using the previously saved checkpoint.  By validating that
// the device values return the initial values, this demonstrates that the
// checkpoint API worked as expected.

// System headers
#include <iostream>
#include <stdlib.h>
using namespace std;

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include <cupti_checkpoint.h>
#include "helper_cupti.h"
using namespace NV::Cupti::Checkpoint;

// Kernels
// Basic example of a kernel which may overwrite its own input data
// This is not intended to show how to write a well-designed reduction,
// but to demonstrate that a kernel which modifies its input data can be
// replayed using the checkpoint API and get reproducible results.
//
// Sums N values, returning the total sum in pData[0]
__global__ void
Reduce(
    float *pData,
    size_t N)
{
    float totalSumData = 0.0;

    // Each thread sums its elements locally.
    for (int i = threadIdx.x; i < N; i+= blockDim.x)
    {
        totalSumData += pData[i];
    }

    // And saves the per-thread sum back to the thread's first element.
    pData[threadIdx.x] = totalSumData;

    __syncthreads();

    // Then, thread 0 reduces those per-thread sums to a single value in pData[0].
    if (threadIdx.x == 0)
    {
        float totalSum = 0.0;

        size_t setElements = (blockDim.x < N ? blockDim.x : N);

        for (int i = 0; i < setElements; i++)
        {
            totalSum += pData[i];
        }

        pData[0] = totalSum;
    }
}

// Functions
int
main()
{
    CUcontext ctx;

    // Set up a context for device 0.
    RUNTIME_API_CALL(cudaSetDevice(0));
    DRIVER_API_CALL(cuCtxCreate(&ctx, 0, 0));

    // Allocate host and device arrays and initialize to known values.
    float *pDeviceA;
    size_t elements = 1024 * 1024;
    size_t size = elements * sizeof(float);

    RUNTIME_API_CALL(cudaMalloc(&pDeviceA, size));
    MEMORY_ALLOCATION_CALL(pDeviceA);

    float *pHostA = (float *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    for (size_t i = 0; i < elements; i++)
    {
        pHostA[i] = 1.0;
    }
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));

    cout << "Initially, pDeviceA[0] = " << pHostA[0] << endl;

    // Demonstrate a case where calling a kernel repeatedly may cause incorrect
    // behavior due to internally modifying its input data.
    cout << "Without checkpoint:" << endl;
    for (int repeat = 0; repeat < 3; repeat++)
    {
        Reduce <<< 1, 64 >>> (pDeviceA, elements);

        // Test return value - should change each iteration due to not resetting input array.
        float ret;
        RUNTIME_API_CALL(cudaMemcpy(&ret, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost));
        cout << "After " << (repeat + 1) << " iteration" << (repeat > 0 ? "s" : "") << ", pDeviceA[0] = " << ret << endl;
    }

    // Re-initialize input array.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    cout << "Reset device array - pDeviceA[0] = " << pHostA[0] << endl;

    // Configure a checkpoint object.
    CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
    cp.ctx = ctx;
    cp.optimizations = 1;

    float expected;

    cout << "With checkpoint:" << endl;
    for (int repeat = 0; repeat < 3; repeat++)
    {
        // Save or restore the checkpoint as needed.
        if (repeat == 0)
        {
            CUPTI_API_CALL(cuptiCheckpointSave(&cp));
        }
        else
        {
            CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
        }

        // Call reduction kernel that modifies its own input.
        Reduce <<< 1, 64 >>> (pDeviceA, elements);

        // Check the output value. (pDeviceA[0])
        float ret;
        RUNTIME_API_CALL(cudaMemcpy(&ret, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost));

        // The first call to the kernel produces the expected result - with checkpoint, every subsequent call should also return this.
        if (repeat == 0)
        {
            expected = ret;
        }

        cout << "After " << (repeat + 1) << " iteration" << (repeat > 0 ? "s" : "") << ", pDeviceA[0] = " << ret << endl;

        // Verify that this iteration's output value matches the expected value from the first iteration.
        if (ret != expected)
        {
            cerr << "Error - repeat " << repeat << " did not match expected value (" << ret << " != " << expected << "), did checkpoint not restore input data correctly?" << endl;
            CUPTI_API_CALL(cuptiCheckpointFree(&cp));

            exit(EXIT_FAILURE);
        }
    }

    // Clean up.
    CUPTI_API_CALL(cuptiCheckpointFree(&cp));

    exit(EXIT_SUCCESS);
}
