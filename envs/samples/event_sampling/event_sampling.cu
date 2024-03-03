/*
 * Copyright 2011-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler
 * event values by sampling.
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// System headers
#include <stdio.h>
#include <stdlib.h>

// CUDA headers
#include <cuda_runtime_api.h>

// CUPTI headers
#include <cupti_events.h>
#include "helper_cupti.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#endif

// Macros
#define EVENT_NAME "inst_executed"
#define N 100000
#define ITERATIONS 10000
#define SAMPLE_PERIOD_MS 50

// Global variables
#ifdef _WIN32
HANDLE semaphore;
DWORD ret;
#else
sem_t semaphore;
int ret;
#endif

// Used to signal from the compute thread to the sampling thread.
static volatile int testComplete = 0;

static CUcontext context;
static CUdevice device;
static const char *s_pEventName;

// Kernels
__global__ void
VectorAdd(
    const int *pA,
    const int *pB,
    int *pC,
    int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int n = 0 ; n < 100; n++)
    {
        if (i < size)
        {
            pC[i] = pA[i] + pB[i];
        }
    }
}

// Functions
static void
InitializeVector(
    int *pVector,
    int n)
{
    for (int i = 0; i < n; i++)
    {
        pVector[i] = i;
    }
}

void *
DoSampling(
    void *arg)
{
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    size_t bytesRead, valueSize;
    uint32_t numInstances = 0, j = 0;
    uint64_t *pEventValues = NULL, eventValue = 0;
    uint32_t profileAll = 1;

    CUPTI_API_CALL(cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

    CUPTI_API_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));

    CUPTI_API_CALL(cuptiEventGetIdFromName(device, s_pEventName, &eventId));

    CUPTI_API_CALL(cuptiEventGroupAddEvent(eventGroup, eventId));

    CUPTI_API_CALL(cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profileAll), &profileAll));

    CUPTI_API_CALL(cuptiEventGroupEnable(eventGroup));

    valueSize = sizeof(numInstances);
    CUPTI_API_CALL(cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &valueSize, &numInstances));

    bytesRead = sizeof(uint64_t) * numInstances;
    pEventValues = (uint64_t *)malloc(bytesRead);
    MEMORY_ALLOCATION_CALL(pEventValues);

    // Release the semaphore as sampling thread is ready to read events.
#ifdef _WIN32
    ret = ReleaseSemaphore(semaphore, 1, NULL);
    if (ret == 0)
    {
        printf("Error: Failed to release the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#else
    ret = sem_post(&semaphore);
    if (ret != 0)
    {
        printf("Error: Failed to release the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#endif

    do
    {
        CUPTI_API_CALL(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, eventId, &bytesRead, pEventValues));

        if (bytesRead != (sizeof(uint64_t) * numInstances))
        {
            printf("Error: Failed to read value for \"%s\"\n", s_pEventName);
            exit(EXIT_FAILURE);
        }

        for (j = 0; j < numInstances; j++)
        {
            eventValue += pEventValues[j];
        }
        printf("%s: %llu\n", s_pEventName, (unsigned long long)eventValue);

#ifdef _WIN32
        Sleep(SAMPLE_PERIOD_MS);
#else
        usleep(SAMPLE_PERIOD_MS * 1000);
#endif
    }
    while (!testComplete);

    CUPTI_API_CALL(cuptiEventGroupDisable(eventGroup));

    CUPTI_API_CALL(cuptiEventGroupDestroy(eventGroup));

    free(pEventValues);

    return NULL;
}

static void
DoCompute(
    int iters)
{
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int sum, i;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors.
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Invoke kernel. (Multiple times to make sure we have time for
    // sampling.)
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (i = 0; i < iters; i++)
    {
        VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        sum = pHostA[i] + pHostB[i];
        if (pHostC[i] != sum)
        {
            printf("Error: Kernel execution failed.\n");
            exit(EXIT_FAILURE);
        }
    }

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
}

int
main(int argc, char *argv[])
{
#ifdef _WIN32
    HANDLE hThread;
#else
    int status;
    pthread_t pThread;
#endif
    int deviceNum;
    int deviceCount;
    char deviceName[256];
    int major;
    int minor;

    printf("Usage: %s [device_num] [event_name]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Error: There is no device supporting CUDA.\n");
        exit(EXIT_WAIVED);
    }

    if (argc > 1)
    {
        deviceNum = atoi(argv[1]);
    }
    else
    {
        deviceNum = 0;
    }
    printf("CUDA Device Number: %d\n", deviceNum);

    DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
    printf("CUDA Device Name: %s\n", deviceName);

    DRIVER_API_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    DRIVER_API_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("Compute Capability of Device: %d.%d\n", major,minor);

    int deviceComputeCapability = 10 * major + minor;
    if (deviceComputeCapability > 72)
    {
        printf("Warning: Sample unsupported on Device with compute capability > 7.2\n");
        exit(EXIT_WAIVED);
    }

    if (argc > 2)
    {
        s_pEventName = argv[2];
    }
    else
    {
        s_pEventName = EVENT_NAME;
    }

    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

    // Create semaphore.
#ifdef _WIN32
    semaphore = CreateSemaphore(NULL, 0, 10, NULL);
    if (semaphore == NULL)
    {
        printf("Error: Failed to create the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#else
    ret = sem_init(&semaphore, 0, 0);
    if (ret != 0) {
        printf("Error: Failed to create the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#endif

    testComplete = 0;

    printf("Creating sampling thread\n");
#ifdef _WIN32
    hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) DoSampling, NULL, 0, NULL );
    if (!hThread)
    {
        printf("Error: CreateThread failed.\n");
        exit(EXIT_FAILURE);
    }
#else
    status = pthread_create(&pThread, NULL, DoSampling, NULL);
    if (status != 0)
    {
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }
#endif

    // Wait for sampling thread to be ready for event collection
#ifdef _WIN32
    ret = WaitForSingleObject(semaphore, INFINITE);
    if (ret != WAIT_OBJECT_0)
    {
        printf("Error: Failed to wait for the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#else
    ret = sem_wait(&semaphore);
    if (ret != 0)
    {
        printf("Error: Failed to wait for the semaphore.\n");
        exit(EXIT_FAILURE);
    }
#endif

    // Run kernel while sampling.
    DoCompute(ITERATIONS);

    // "signal" the sampling thread to exit and wait for it.
    testComplete = 1;
#ifdef _WIN32
    WaitForSingleObject(hThread, INFINITE);
#else
    pthread_join(pThread, NULL);
#endif

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    exit(EXIT_SUCCESS);
}
