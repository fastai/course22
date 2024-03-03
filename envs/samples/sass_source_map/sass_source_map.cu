/*
 * Copyright 2014-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print sass to source correlation
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Macros
#define DUMP_CUBIN 0

// Global variables
const int TileDim   = 32;
const int BlockRows = 8;

// Kernels
__global__
void Transpose(
    float *pOutputData,
    const float *pInputData)
{
    __shared__ float tile[TileDim][TileDim+1];

    int x = blockIdx.x * TileDim + threadIdx.x;
    int y = blockIdx.y * TileDim + threadIdx.y;
    int width = gridDim.x * TileDim;

    for (int j = 0; j < TileDim; j += BlockRows)
    {
        tile[threadIdx.y + j][threadIdx.x] = pInputData[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TileDim + threadIdx.x;
    y = blockIdx.x * TileDim + threadIdx.y;

    for (int j = 0; j < TileDim; j += BlockRows)
    {
        pOutputData[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// Functions
static void
DumpCudaModule(
    CUpti_CallbackId callbackId,
    void *pResourceDescriptor)
{
#if DUMP_CUBIN
    const char *pCubin;
    size_t cubinSize;

    // Dump the cubin at MODULE_LOADED_STARTING.
    CUpti_ModuleResourceData *pModuleResourceData = (CUpti_ModuleResourceData *)pResourceDescriptor;
#endif

    if (callbackId == CUPTI_CBID_RESOURCE_MODULE_LOADED)
    {
#if DUMP_CUBIN
        // You can use nvdisasm to dump the SASS from the cubin.
        // Try nvdisasm -b -fun <function_id> sass_to_source.cubin
        pCubin    = pModuleResourceData->pCubin;
        cubinSize = pModuleResourceData->cubinSize;

        FILE *pCubinFileHandle;
        pCubinFileHandle = fopen("sass_source_map.cubin", "wb");
        fwrite(pCubin, sizeof(uint8_t), cubinSize, pCubinFileHandle);
        fclose(pCubinFileHandle);
#endif
    }
    else if (callbackId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING)
    {
        // You can dump the cubin either at MODULE_LOADED or MODULE_UNLOAD_STARTING.
    }
}

static void
HandleResource(
    CUpti_CallbackId callbackId,
    const CUpti_ResourceData *pResourceData)
{
    if (callbackId == CUPTI_CBID_RESOURCE_MODULE_LOADED)
    {
        DumpCudaModule(callbackId, pResourceData->resourceDescriptor);
    }
    else if (callbackId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING)
    {
        DumpCudaModule(callbackId, pResourceData->resourceDescriptor);
    }
}

void CUPTIAPI
TraceCallback(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const void *CallbackData)
{
    // Check last error.
    CUPTI_API_CALL(cuptiGetLastError());

    if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        HandleResource(callbackId, (CUpti_ResourceData *)CallbackData);
    }
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, (void *)TraceCallback, stdout);

    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
    CUPTI_API_CALL(cuptiEnableDomain(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE));
}

int
main(
    int argc,
    char *argv[])
{
    const int Nx = 32;
    const int Ny = 32;
    const int MemSize = Nx * Ny * sizeof(float);

    dim3 dimGrid(Nx/TileDim, Ny/TileDim, 1);
    dim3 dimBlock(TileDim, BlockRows, 1);

    cudaDeviceProp deviceProperties;

    SetupCupti();

    RUNTIME_API_CALL(cudaGetDeviceProperties(&deviceProperties, 0));
    printf("Device Name: %s\n", deviceProperties.name);

    float *pDeviceA, *pDeviceB;

    float *pHostA = (float *)malloc(MemSize);
    MEMORY_ALLOCATION_CALL(pHostA);

    float *pHostB = (float *)malloc(MemSize);
    MEMORY_ALLOCATION_CALL(pHostB);

    // Initialization of host data.
    for (int j = 0; j < Ny; j++)
    {
        for (int i = 0; i < Nx; i++)
        {
            pHostA[ (j * Nx) + i] = (float) ((j * Nx) + i);
        }
    }
    RUNTIME_API_CALL(cudaMalloc(&pDeviceA, MemSize));
    RUNTIME_API_CALL(cudaMalloc(&pDeviceB, MemSize));

    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, MemSize, cudaMemcpyHostToDevice));

    Transpose <<< dimGrid, dimBlock >>> (pDeviceB, pDeviceA);
    RUNTIME_API_CALL(cudaGetLastError());

    RUNTIME_API_CALL(cudaMemcpy(pHostB, pDeviceB, MemSize, cudaMemcpyDeviceToHost));

    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }

    RUNTIME_API_CALL(cudaDeviceSynchronize());
    RUNTIME_API_CALL(cudaDeviceReset());

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}

