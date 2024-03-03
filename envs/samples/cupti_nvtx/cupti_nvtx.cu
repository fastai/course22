/*
 * Copyright 2021-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to output NVTX ranges.
 * The sample adds NVTX ranges around a simple vector addition app
 * NVTX functionality shown in the sample:
 *  Subscribe to NVTX callbacks and get NVTX records
 *  Create domain, add start/end and push/pop ranges w.r.t the domain
 *  Register string against a domain
 *  Naming of CUDA resources
 *
 * Before running the sample set the NVTX_INJECTION64_PATH
 * environment variable pointing to the CUPTI Library.
 * For Linux:
 *    export NVTX_INJECTION64_PATH=<full_path>/libcupti.so
 * For Windows:
 *    set NVTX_INJECTION64_PATH=<full_path>/cupti.dll
 */

// System headers
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include <helper_cupti_activity.h>

// NVTX headers
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

// Includes definition of the callback structures to use for NVTX with CUPTI.
#include "generated_nvtx_meta.h"

__global__ void
VectorAdd(
    const int *pA,
    const int *pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

static void
InitializeVector(
    int *pVector,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

void
DoVectorAddition()
{
    CUcontext context = 0;
    CUdevice device = 0;
    int N = 50000;
    size_t size = N * sizeof (int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *pHostA = 0, *pHostB = 0, *pHostC = 0;
    int *pDeviceA = 0, *pDeviceB = 0, *pDeviceC = 0;

    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    nvtxNameCuDeviceA(device, "CUDA Device 0");

    // Create domain "Vector Addition".
    nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x0000ff;

    // Push range "doPass" on domain "Vector Addition".
    eventAttrib.message.ascii = "vectorAdd";
    nvtxDomainRangePushEx(domain, &eventAttrib);


    // Push range "Allocate host memory" on default domain.
    nvtxRangePushA("Allocate host memory");

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Pop range "Allocate host memory" on default domain.
    nvtxRangePop();

    // Initialize input vectors.
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    nvtxNameCuContextA(context, "CUDA Context");

    // Push range "Allocate device memory" on domain "Vector Addition".
    eventAttrib.message.ascii = "Allocate device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Pop range on domain - Allocate device memory.
    nvtxDomainRangePop(domain);

    // Register string "Memcpy operation".
    nvtxStringHandle_t string = nvtxDomainRegisterStringA(domain, "Memcpy operation");
    // Push range "Memcpy operation" on domain "Vector Addition".
    eventAttrib.message.registered = string;
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Push range "Memcpy operation" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Start range "Launch kernel" on domain "Vector Addition".
    eventAttrib.message.ascii = "Launch kernel";
    nvtxRangeId_t id = nvtxDomainRangeStartEx(domain, &eventAttrib);

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());
    DRIVER_API_CALL(cuCtxSynchronize());

    // End range "Launch kernel" on domain "Vector Addition".
    nvtxDomainRangeEnd(domain, id);

    eventAttrib.message.registered = string;
    // Push range "Memcpy operation" on domain "Vector Addition".
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Push range "Memcpy operation" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Push range "Free device memory" on domain "Vector Addition".
    eventAttrib.message.ascii = "Free device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }
    if (pDeviceC)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }

    // Push range "Free device memory" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Push range "Free host memory" on default domain.
    nvtxRangePushA("Free host memory");

    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }
    if (pHostC)
    {
        free(pHostC);
    }

    // Pop range "Free host memory" on default domain.
    nvtxRangePop();

    DRIVER_API_CALL(cuCtxSynchronize());
    DRIVER_API_CALL(cuCtxDestroy(context));

    // Pop range "vectorAdd" on domain "Vector Addition".
    nvtxDomainRangePop(domain);
}

void CUPTIAPI
NvtxCallback(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const void *pCallbackData)
{
    // Commented the NVTX code to avoid warnings for unused varaibles.
    // CUpti_NvtxData* pNvtxData = (CUpti_NvtxData*)pCallbackData;

    switch (callbackId) {
        case CUPTI_CBID_NVTX_nvtxDomainCreateA:
        {
            // Get the parameters passed to the NVTX function.
            // nvtxDomainCreateA_params* params = (nvtxDomainCreateA_params*)pNvtxData->functionParams;
            // Get the return value of the NVTX function.
            // nvtxDomainHandle_t* domainHandle = (nvtxDomainHandle_t*)pNvtxData->functionReturnValue;
            break;
        }
        case CUPTI_CBID_NVTX_nvtxMarkEx:
        {
            // nvtxMarkEx_params* params = (nvtxMarkEx_params*)pNvtxData->functionParams;
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx:
        {
            // nvtxDomainMarkEx_params* params = (nvtxDomainMarkEx_params*)pNvtxData->functionParams;
            break;
        }
        // Add more NVTX callbacks, refer "generated_nvtx_meta.h" for all NVTX callbacks.
        // If there is no return value for the NVTX function, functionReturnValue is NULL.
        default:
            break;
    }

    return;
}

void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, (void *)NvtxCallback, stdout);

    CUPTI_API_CALL(cuptiEnableDomain(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_NVTX));

    // For NVTX markers. (Marker, Domain, Start/End ranges, Push/Pop ranges, Registered Strings)
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    // For naming CUDA resources. (Threads, Devices, Contexts, Streams)
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
}

int
main(
    int argc,
    char *argv[])
{
    SetupCupti();

    // Initialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    DoVectorAddition();

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);
}
