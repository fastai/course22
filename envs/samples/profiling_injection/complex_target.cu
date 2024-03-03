// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// This is a sample CUDA application with several different kernel launch
// patterns - launching on the default stream, multple streams, and multiple
// threads on different devices, if more than one device is present.
//
// The injection sample shared library can be used on this sample application,
// demonstrating that the injection code handles multple streams and multiple
// threads.

// CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "helper_cupti.h"

// Standard STL headers
#include <chrono>
#include <cstdint>
#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <string>
using ::std::string;

#include <thread>
using ::std::thread;

#include <vector>
using ::std::vector;

#include <stdlib.h>

// Per-device configuration, buffers, stream and device information, and device pointers.
typedef struct PerDeviceData_st
{
    int deviceId;
    // CUDA driver context, or NULL if default context has already been initialized.
    CUcontext context;
    // Each device needs its own streams.
    vector<cudaStream_t> streams;
    // And device memory allocation.
    vector<double *> pDeviceX;
    vector<double *> pDeviceY;
} PerDeviceData;

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times.
// To be launched with only one block. (Artificially increasing serial time to better demonstrate overlapping replay.)
__global__ void
DaxpyKernel(
    int elements,
    double a,
    double *pX,
    double *pY)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
    {
        // Artificially increase kernel runtime to emphasize concurrency.
        for (int j = 0; j < DAXPY_REPEAT; j++)
        {
            // daxpy
            pY[i] = a * pX[i] + pY[i];
        }
    }
}

// Initializekernel values.
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels. (streams, when running concurrently.)
int const numKernels = 4;
int const numStreams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements.
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB.
int const blockSize = 4 * 1024;

// Wrapper which will launch numKernel kernel calls on a single device.
// The device streams vector is used to control which stream each call is made on.
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used.
void
LaunchKernels(
    PerDeviceData &deviceData,
    char const * const pRangeName,
    bool serial)
{
    // Switch to desired device.
    RUNTIME_API_CALL(cudaSetDevice(deviceData.deviceId));
    DRIVER_API_CALL(cuCtxSetCurrent(deviceData.context));

    for (unsigned int stream = 0; stream < deviceData.streams.size(); stream++)
    {
        cudaStream_t streamId = (serial ? 0 : deviceData.streams[stream]);
        DaxpyKernel <<< threadBlocks, threadsPerBlock, 0, streamId >>> (elements[stream], a, deviceData.pDeviceX[stream], deviceData.pDeviceY[stream]);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    // After launching all work, synchronize all streams.
    if (serial == false)
    {
        for (unsigned int stream = 0; stream < deviceData.streams.size(); stream++)
        {
            RUNTIME_API_CALL(cudaStreamSynchronize(deviceData.streams[stream]));
        }
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
    }
}


int
main(
    int argc,
    char *argv[])
{
    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information.
    vector<int> deviceIds;

    // Find all devices.
    for (int i = 0; i < numDevices; i++)
    {
        // Record device number.
        deviceIds.push_back(i);
    }

    numDevices = deviceIds.size();
    cout << "Found " << numDevices << " devices" << endl;

    // Ensure we found at least one device.
    if (numDevices == 0)
    {
        cerr << "No devices detected" << endl;
        exit(EXIT_WAIVED);
    }

    // Initialize kernel input to some known numbers.
    vector<double> hX(blockSize * numKernels);
    vector<double> hY(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        hX[i] = 1.5 * i;
        hY[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels.
    vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++)
    {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number.
    for (int stream = 0; stream < numStreams; stream++)
    {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data.
    vector<PerDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        RUNTIME_API_CALL(cudaSetDevice(deviceIds[device]));
        cout << "Configuring device " << deviceIds[device] << endl;

        // For simplicity's sake, in this sample, a single config struct is created per device.
        deviceData[device].deviceId = deviceIds[device];// GPU device ID

        // Either set to a context, or may be NULL if a default context has been created.
        DRIVER_API_CALL(cuCtxCreate(&(deviceData[device].context), 0, deviceIds[device]));

        // Per-stream initialization & memory allocation - copy from constant host array to each device array.
        deviceData[device].streams.resize(numStreams);
        deviceData[device].pDeviceX.resize(numStreams);
        deviceData[device].pDeviceY.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles.
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].pDeviceX[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].pDeviceX[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].pDeviceX[stream], hX.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].pDeviceY[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].pDeviceY[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].pDeviceY[stream], hX.data(), size, cudaMemcpyHostToDevice));
        }
    }

    // First version - single device, kernel calls serialized on default stream.
    // Use wallclock time to measure performance.
    auto beginTime = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams - will show runtime without any concurrency.
    LaunchKernels(deviceData[0], "single_gpu_serial", true);

    auto endTime = ::std::chrono::high_resolution_clock::now();
    auto elapsedSerialMs = ::std::chrono::duration_cast<::std::chrono::milliseconds>(endTime - beginTime);
    cout << "It took " << elapsedSerialMs.count() << "ms on the host to launch " << numKernels << " kernels in serial" << endl;

    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency.
    // (Should be limited by the longest running kernel.)
    beginTime = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism.
    LaunchKernels(deviceData[0], "single_gpu_async", false);

    endTime = ::std::chrono::high_resolution_clock::now();
    auto elapsedSingleDeviceMs = ::std::chrono::duration_cast<::std::chrono::milliseconds>(endTime - beginTime);
    cout << "It took " << elapsedSingleDeviceMs.count() << "ms on the host to launch " << numKernels << " kernels on a single device on separate streams" << endl;

    // Third version - same as the second case, but duplicate the work across devices to show cross-device concurrency.
    // This is done using threads so no serialization is needed between devices.
    // (Should have roughly the same runtime as second case.)
    // Time creation of the same multiple streams * multiple devices.
    vector<::std::thread> threads;
    beginTime = ::std::chrono::high_resolution_clock::now();

    // Now launch parallel thread work, duplicated on one thread per GPU.
    for (int device = 0; device < numDevices; device++)
    {
        threads.push_back(::std::thread(LaunchKernels, ::std::ref(deviceData[device]), "multi_gpu_async", false));
    }

    // Wait for all threads to finish.
    for (auto &t: threads)
    {
        t.join();
    }

    // Record time used when launching on multiple devices.
    endTime = ::std::chrono::high_resolution_clock::now();
    auto elapsedMultipleDeviceMs = ::std::chrono::duration_cast<::std::chrono::milliseconds>(endTime - beginTime);
    cout << "It took " << elapsedMultipleDeviceMs.count() << "ms on the host to launch the same " << numKernels << " kernels on each of the " << numDevices << " devices in parallel" << endl;

    // Free stream memory for each device.
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].pDeviceX[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].pDeviceY[j]));
        }
    }

    exit(EXIT_SUCCESS);
}
