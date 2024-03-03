/*
 * Copyright 2020-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of pc sampling APIs.
 * This app will work on devices with compute capability 7.0 and higher.
 *
 * Work flow in brief:
 *
 *    Subscribed for all the launch callbacks and required resource callbacks like module and context callbacks
 *        Context created callback:
 *            Enable PC sampling using cuptiPCSamplingEnable() CUPTI API.
 *            Configure PC sampling for that context in ConfigureActivity() function.
 *                ConfigureActivity():
 *                    Get count of all stall reasons supported on GPU using cuptiPCSamplingGetNumStallReasons() CUPTI API.
 *                    Get all stall reasons names and its indexes using cuptiPCSamplingGetStallReasons() CUPTI API.
 *                    Configure PC sampling with provide parameters and to sample all stall reasons using
 *                    cuptiPCSamplingSetConfigurationAttribute() CUPTI API.
 *            Only for first context creation, create worker thread which will store flushed buffers from the
 *            queue of buffers into the file.
 *            Only for first context creation, allocate memory for circular buffers which will hold flushed data from cupti.
 *
 *        Launch callbacks:
 *           If serialized mode is enabled then every time if cupti has PC records then flush all records using
 *           cuptiPCSamplingGetData() and push buffer in queue with context info to store it in file.
 *           If continuous mode is enabled then if cupti has more records than size of single circular buffer
 *           then flush records in one circular buffer using cuptiPCSamplingGetData() and push it in queue with
 *           context info to store it in file.
 *
 *        Module load:
 *           This callback covers case when module get unloaded and new module get loaded then cupti flush
 *           all records into the provided buffer during configuration.
 *           So in this callback if provided buffer during configuration has any records then flush all records into
 *           the circular buffers and push them into the queue with context info to store them into the file.
 *
 *        Context destroy starting:
 *           Disable PC sampling using cuptiPCSamplingDisable() CUPTI API
 *
 *    AtExitHandler
 *        If PC sampling is not disabled for any context then disable it using cuptiPCSamplingDisable().
 *        Push PC sampling buffer in queue which provided during configuration with context info for each context
 *        as cupti flush all remaining PC records into this buffer in the end.
 *        Join the thread after storing all buffers present in the queue.
 *        Free allocated memory for circular buffer, stall reason names, stall reasons indexes and
 *        PC sampling buffers provided during configuration.
 *
 *    Worker thread:
 *        Worker thread read front of queue take buffer and from context info read context id to store data into
 *        the file <context_id>_<file name>. Also it read configuration info and stall reason info from context info
 *        and store it in file using CuptiUtilPutPcSampData() CUPTI PC sampling Util API.
 *        Worker thread stores all buffers till the queue gets empty and then goes to sleep.
 *        It got joined to the main thread in AtExitHandler.
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <inttypes.h>
#include <unordered_set>
#include <mutex>
#include <map>
#include <queue>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include "detours.h"
#else
#include <unistd.h>
#include <pthread.h>
#endif

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include <cupti_pcsampling_util.h>
#include <cupti_pcsampling.h>
#include "helper_cupti.h"
#include <cupti.h>
using namespace CUPTI::PcSamplingUtil;

// Macros
#define THREAD_SLEEP_TIME 100 // in ms

// Global structures and variables
typedef struct ContextInfo_st
{
    uint32_t contextUid;
    CUpti_PCSamplingData pcSamplingData;
    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
    PcSamplingStallReasons pcSamplingStallReasons;
} ContextInfo;

// For multi-gpu we are preallocating buffers only for first context creation,
// So preallocated buffer stall reason size will be equal to max stall reason for first context GPU.
size_t stallReasonsCount = 0;
// Consider firstly queried stall reason count using cuptiPCSamplingGetNumStallReasons() to allocate memory for circular buffers.
bool g_collectedStallReasonsCount = false;
std::mutex g_stallReasonsCountMutex;

// Variables related to circular buffer.
std::vector<CUpti_PCSamplingData> g_circularBuffer;
std::unordered_set<char*> functions;
int g_put = 0;
int g_get = 0;
std::vector<bool> g_bufferEmptyTrackerArray; // true - used, false - free.
std::mutex g_circularBufferMutex;
bool g_buffersGetUtilisedFasterThanStore = false;
bool g_allocatedCircularBuffers = false;

// Variables related to context info book keeping.
std::map<CUcontext, ContextInfo *> g_contextInfoMap;
std::mutex g_contextInfoMutex;
std::vector<ContextInfo *> g_contextInfoToFreeInEndVector;

// Variables related to thread which store data in file.
std::string g_fileName = "pcsampling.dat";
std::thread g_storeDataInFileThreadHandle;
std::queue<std::pair<CUpti_PCSamplingData *, ContextInfo *>> g_pcSampDataQueue;
bool g_waitAtJoin = false;
std::mutex g_pcSampDataQueueMutex;
bool g_createdWorkerThread = false;
std::mutex g_workerThreadMutex;

// Variables related to initialize injection once.
bool g_initializedInjection = false;
std::mutex g_initializeInjectionMutex;

// Variables for args set through script.
CUpti_PCSamplingCollectionMode g_pcSamplingCollectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
uint32_t g_samplingPeriod = 0;
size_t g_scratchBufSize = 0;
size_t g_hwBufSize = 0;
size_t g_pcConfigBufRecordCount = 5000;
size_t g_circularbufCount = 10;
size_t g_circularbufSize = 500;
bool g_disableFileDump = false;
bool g_verbose = false;

bool g_running = false;

static void
ReadInputParams()
{
    char *pInjectionParam = getenv("INJECTION_PARAM");
    if (pInjectionParam == NULL)
    {
        g_circularBuffer.resize(g_circularbufCount);
        g_bufferEmptyTrackerArray.resize(g_circularbufCount, false);
        return;
    }

    char *pToken = strtok(pInjectionParam, " ");

    while (pToken != NULL)
    {
        if (!strcmp(pToken, "--collection-mode"))
        {
            pToken = strtok(NULL," ");
            g_pcSamplingCollectionMode = (CUpti_PCSamplingCollectionMode)atoi(pToken);
        }
        else if (!strcmp(pToken, "--sampling-period"))
        {
            pToken = strtok(NULL," ");
            g_samplingPeriod = (uint32_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--scratch-buf-size"))
        {
            pToken = strtok(NULL," ");
            g_scratchBufSize = (size_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--hw-buf-size"))
        {
            pToken = strtok(NULL," ");
            g_hwBufSize = (size_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--pc-config-buf-record-count"))
        {
            pToken = strtok(NULL," ");
            g_pcConfigBufRecordCount = (size_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--pc-circular-buf-record-count"))
        {
            pToken = strtok(NULL," ");
            g_circularbufSize = (size_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--circular-buf-count"))
        {
            pToken = strtok(NULL," ");
            g_circularbufCount = (size_t)atoi(pToken);
        }
        else if (!strcmp(pToken, "--file-name"))
        {
            pToken = strtok(NULL," ");
            std::string file(pToken);
            g_fileName = file;
        }
        else if (!strcmp(pToken, "--disable-file-dump"))
        {
            g_disableFileDump = true;
        }
        else if (!strcmp(pToken, "--verbose"))
        {
            g_verbose = true;
        }
        pToken = strtok(NULL," ");
    }
    g_circularBuffer.resize(g_circularbufCount);
    g_bufferEmptyTrackerArray.resize(g_circularbufCount, false);
}

static bool
GetPcSamplingDataFromCupti(
    CUpti_PCSamplingGetDataParams &pcSamplingGetDataParams,
    ContextInfo *pContextInfo)
{
    CUpti_PCSamplingData *pPcSamplingData = NULL;

    g_circularBufferMutex.lock();
    while (g_bufferEmptyTrackerArray[g_put])
    {
        g_buffersGetUtilisedFasterThanStore = true;
    }

    pcSamplingGetDataParams.pcSamplingData = (void *)&g_circularBuffer[g_put];
    pPcSamplingData = &g_circularBuffer[g_put];

    if (!g_disableFileDump)
    {
        g_bufferEmptyTrackerArray[g_put] = true;
        g_put = (g_put + 1) % g_circularbufCount;
    }
    g_circularBufferMutex.unlock();

    CUptiResult cuptiStatus = cuptiPCSamplingGetData(&pcSamplingGetDataParams);
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        CUpti_PCSamplingData *samplingData = (CUpti_PCSamplingData*)pcSamplingGetDataParams.pcSamplingData;
        if (samplingData->hardwareBufferFull)
        {
            printf("ERROR!! hardware buffer is full, need to increase hardware buffer size or frequency of pc sample data decoding\n");
            return false;
        }
    }

    if (!g_disableFileDump)
    {
        g_pcSampDataQueueMutex.lock();
        g_pcSampDataQueue.push(std::make_pair(pPcSamplingData, pContextInfo));
        g_pcSampDataQueueMutex.unlock();
    }

    return true;
}

static void
StorePcSampDataInFile()
{
    CUptiUtilResult utilResult;
    ContextInfo *pContextInfo;
    CUpti_PCSamplingData *pcSamplingData;

    g_pcSampDataQueueMutex.lock();
    pcSamplingData = g_pcSampDataQueue.front().first;
    pContextInfo = g_pcSampDataQueue.front().second;
    g_pcSampDataQueue.pop();
    g_pcSampDataQueueMutex.unlock();

    std::string file = std::to_string((long int)pContextInfo->contextUid) + "_" + g_fileName;

    CUptiUtil_PutPcSampDataParams pPutPcSampDataParams = {};
    pPutPcSampDataParams.size = CUptiUtil_PutPcSampDataParamsSize;
    pPutPcSampDataParams.bufferType = PC_SAMPLING_BUFFER_PC_TO_COUNTER_DATA;
    pPutPcSampDataParams.pSamplingData = (void*)pcSamplingData;
    pPutPcSampDataParams.numAttributes = pContextInfo->pcSamplingConfigurationInfo.size();
    pPutPcSampDataParams.pPCSamplingConfigurationInfo = pContextInfo->pcSamplingConfigurationInfo.data();
    pPutPcSampDataParams.pPcSamplingStallReasons = &pContextInfo->pcSamplingStallReasons;
    pPutPcSampDataParams.fileName = file.c_str();

    utilResult = CuptiUtilPutPcSampData(&pPutPcSampDataParams);
    if (utilResult != CUPTI_UTIL_SUCCESS)
    {
        std::cout << "error in StorePcSampDataInFile(), failed with error : " << utilResult << std::endl;
        exit (EXIT_FAILURE);
    }
    for (size_t i = 0; i < pcSamplingData->totalNumPcs; i++)
    {
        functions.insert(pcSamplingData->pPcData[i].functionName);
    }
    g_bufferEmptyTrackerArray[g_get] = false;
    g_get = (g_get + 1) % g_circularbufCount;
}

static void
StorePcSampDataInFileThread()
{
    while (1)
    {
        if (g_waitAtJoin)
        {
            while (!g_pcSampDataQueue.empty())
            {
                StorePcSampDataInFile();
            }
            break;
        }
        else
        {
            while (!g_pcSampDataQueue.empty())
            {
                StorePcSampDataInFile();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(THREAD_SLEEP_TIME));
        }
    }
}

static void
PreallocateBuffersForRecords()
{
    for (size_t buffers = 0; buffers < g_circularbufCount; buffers++)
    {
        g_circularBuffer[buffers].size = sizeof(CUpti_PCSamplingData);
        g_circularBuffer[buffers].collectNumPcs = g_circularbufSize;
        g_circularBuffer[buffers].pPcData = (CUpti_PCSamplingPCData *)malloc(g_circularBuffer[buffers].collectNumPcs * sizeof(CUpti_PCSamplingPCData));
        MEMORY_ALLOCATION_CALL(g_circularBuffer[buffers].pPcData);
        for (size_t i = 0; i < g_circularBuffer[buffers].collectNumPcs; i++)
        {
            g_circularBuffer[buffers].pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)malloc(stallReasonsCount * sizeof(CUpti_PCSamplingStallReason));
            MEMORY_ALLOCATION_CALL(g_circularBuffer[buffers].pPcData[i].stallReason);
        }
    }
}

static void FreePreallocatedMemory()
{
    for (size_t buffers = 0; buffers < g_circularbufCount; buffers++)
    {
        for (size_t i = 0; i < g_circularBuffer[buffers].collectNumPcs; i++)
        {
            free(g_circularBuffer[buffers].pPcData[i].stallReason);
        }

        free(g_circularBuffer[buffers].pPcData);
    }

    for (auto& itr: g_contextInfoMap)
    {
        // Free PC sampling buffer.
        for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
        {
            free(itr.second->pcSamplingData.pPcData[i].stallReason);
        }
        free(itr.second->pcSamplingData.pPcData);

        for (size_t i = 0; i < itr.second->pcSamplingStallReasons.numStallReasons; i++)
        {
            free(itr.second->pcSamplingStallReasons.stallReasons[i]);
        }
        free(itr.second->pcSamplingStallReasons.stallReasons);
        free(itr.second->pcSamplingStallReasons.stallReasonIndex);

        free(itr.second);
    }

    for(auto& itr: g_contextInfoToFreeInEndVector)
    {
        // Free PC sampling buffer.
        for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
        {
            free(itr->pcSamplingData.pPcData[i].stallReason);
        }
        free(itr->pcSamplingData.pPcData);

        for (size_t i = 0; i < itr->pcSamplingStallReasons.numStallReasons; i++)
        {
            free(itr->pcSamplingStallReasons.stallReasons[i]);
        }
        free(itr->pcSamplingStallReasons.stallReasons);
        free(itr->pcSamplingStallReasons.stallReasonIndex);

        free(itr);
    }

    for (auto it = functions.begin(); it != functions.end(); ++it)
    {
        free(*it);
    }
    functions.clear();
}

void
ConfigureActivity(
    CUcontext cuCtx)
{
    std::map<CUcontext, ContextInfo*>::iterator contextStateMapItr = g_contextInfoMap.find(cuCtx);
    if (contextStateMapItr == g_contextInfoMap.end())
    {
        std::cout << "Error: No context found." << std::endl;
        exit (EXIT_FAILURE);
    }

    CUpti_PCSamplingConfigurationInfo sampPeriod = {};
    CUpti_PCSamplingConfigurationInfo stallReason = {};
    CUpti_PCSamplingConfigurationInfo scratchBufferSize = {};
    CUpti_PCSamplingConfigurationInfo hwBufferSize = {};
    CUpti_PCSamplingConfigurationInfo collectionMode = {};
    CUpti_PCSamplingConfigurationInfo enableStartStop = {};
    CUpti_PCSamplingConfigurationInfo outputDataFormat = {};

    // Get number of supported counters and counter names.
    size_t numStallReasons = 0;
    CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {};
    numStallReasonsParams.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    numStallReasonsParams.ctx = cuCtx;
    numStallReasonsParams.numStallReasons = &numStallReasons;

    g_stallReasonsCountMutex.lock();
    CUPTI_API_CALL(cuptiPCSamplingGetNumStallReasons(&numStallReasonsParams));

    if (!g_collectedStallReasonsCount)
    {
        stallReasonsCount = numStallReasons;
        g_collectedStallReasonsCount = true;
    }
    g_stallReasonsCountMutex.unlock();

    char **pStallReasons = (char **)malloc(numStallReasons * sizeof(char*));
    MEMORY_ALLOCATION_CALL(pStallReasons);
    for (size_t i = 0; i < numStallReasons; i++)
    {
        pStallReasons[i] = (char *)malloc(CUPTI_STALL_REASON_STRING_SIZE * sizeof(char));
        MEMORY_ALLOCATION_CALL(pStallReasons[i]);
    }
    uint32_t *pStallReasonIndex = (uint32_t *)malloc(numStallReasons * sizeof(uint32_t));
    MEMORY_ALLOCATION_CALL(pStallReasonIndex);

    CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {};
    stallReasonsParams.size = CUpti_PCSamplingGetStallReasonsParamsSize;
    stallReasonsParams.ctx = cuCtx;
    stallReasonsParams.numStallReasons = numStallReasons;
    stallReasonsParams.stallReasonIndex = pStallReasonIndex;
    stallReasonsParams.stallReasons = pStallReasons;
    CUPTI_API_CALL(cuptiPCSamplingGetStallReasons(&stallReasonsParams));

    // User buffer to hold collected PC Sampling data in PC-To-Counter format.
    size_t pcSamplingDataSize = sizeof(CUpti_PCSamplingData);
    contextStateMapItr->second->pcSamplingData.size = pcSamplingDataSize;
    contextStateMapItr->second->pcSamplingData.collectNumPcs = g_pcConfigBufRecordCount;
    contextStateMapItr->second->pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)malloc(g_pcConfigBufRecordCount * sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(contextStateMapItr->second->pcSamplingData.pPcData);
    for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
    {
        contextStateMapItr->second->pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)malloc(numStallReasons * sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(contextStateMapItr->second->pcSamplingData.pPcData[i].stallReason);
    }

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;

    stallReason.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
    stallReason.attributeData.stallReasonData.stallReasonCount = numStallReasons;
    stallReason.attributeData.stallReasonData.pStallReasonIndex = pStallReasonIndex;

    CUpti_PCSamplingConfigurationInfo samplingDataBuffer = {};
    samplingDataBuffer.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    samplingDataBuffer.attributeData.samplingDataBufferData.samplingDataBuffer = (void *)&contextStateMapItr->second->pcSamplingData;

    sampPeriod.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    if (g_samplingPeriod)
    {
        sampPeriod.attributeData.samplingPeriodData.samplingPeriod = g_samplingPeriod;
        pcSamplingConfigurationInfo.push_back(sampPeriod);
    }

    scratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    if (g_scratchBufSize)
    {
        scratchBufferSize.attributeData.scratchBufferSizeData.scratchBufferSize = g_scratchBufSize;
        pcSamplingConfigurationInfo.push_back(scratchBufferSize);
    }

    hwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    if (g_hwBufSize)
    {
        hwBufferSize.attributeData.hardwareBufferSizeData.hardwareBufferSize = g_hwBufSize;
        pcSamplingConfigurationInfo.push_back(hwBufferSize);
    }

    collectionMode.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    collectionMode.attributeData.collectionModeData.collectionMode = g_pcSamplingCollectionMode;
    pcSamplingConfigurationInfo.push_back(collectionMode);

    pcSamplingConfigurationInfo.push_back(stallReason);
    pcSamplingConfigurationInfo.push_back(samplingDataBuffer);

    CUpti_PCSamplingConfigurationInfoParams pcSamplingConfigurationInfoParams = {};
    pcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    pcSamplingConfigurationInfoParams.pPriv = NULL;
    pcSamplingConfigurationInfoParams.ctx = cuCtx;
    pcSamplingConfigurationInfoParams.numAttributes = pcSamplingConfigurationInfo.size();
    pcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingConfigurationInfo.data();

    CUPTI_API_CALL(cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    // Store all stall reasons info in context info to dump into the file.
    contextStateMapItr->second->pcSamplingStallReasons.numStallReasons = numStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasons = pStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasonIndex = pStallReasonIndex;

    // Find configuration info and store it in context info to dump in file.
    scratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    hwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    enableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    outputDataFormat.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    outputDataFormat.attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingRetrieveConfigurationInfo;
    pcSamplingRetrieveConfigurationInfo.push_back(collectionMode);
    pcSamplingRetrieveConfigurationInfo.push_back(sampPeriod);
    pcSamplingRetrieveConfigurationInfo.push_back(scratchBufferSize);
    pcSamplingRetrieveConfigurationInfo.push_back(hwBufferSize);
    pcSamplingRetrieveConfigurationInfo.push_back(enableStartStop);

    CUpti_PCSamplingConfigurationInfoParams getPcSamplingConfigurationInfoParams = {};
    getPcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    getPcSamplingConfigurationInfoParams.pPriv = NULL;
    getPcSamplingConfigurationInfoParams.ctx = cuCtx;
    getPcSamplingConfigurationInfoParams.numAttributes = pcSamplingRetrieveConfigurationInfo.size();
    getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingRetrieveConfigurationInfo.data();

    CUPTI_API_CALL(cuptiPCSamplingGetConfigurationAttribute(&getPcSamplingConfigurationInfoParams));

    for (size_t i = 0; i < getPcSamplingConfigurationInfoParams.numAttributes; i++)
    {
        contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[i]);
    }

    contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(outputDataFormat);
    contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(stallReason);

    g_workerThreadMutex.lock();
    if (!g_disableFileDump && !g_createdWorkerThread)
    {
        g_storeDataInFileThreadHandle = std::thread(StorePcSampDataInFileThread);
        g_createdWorkerThread = true;
    }
    g_workerThreadMutex.unlock();

    if (g_verbose)
    {
        std::cout << std::endl;
        std::cout << "============ Configuration Details : ============" << std::endl;
        std::cout << "requested stall reason count : " << numStallReasons << std::endl;
        std::cout << "collection mode              : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[0].attributeData.collectionModeData.collectionMode << std::endl;
        std::cout << "sampling period              : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[1].attributeData.samplingPeriodData.samplingPeriod << std::endl;
        std::cout << "scratch buffer size (Bytes)  : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize << std::endl;
        std::cout << "hardware buffer size (Bytes) : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize << std::endl;
        std::cout << "start stop control           : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[4].attributeData.enableStartStopControlData.enableStartStopControl << std::endl;
        std::cout << "configuration buffer size    : " << g_pcConfigBufRecordCount << std::endl;
        std::cout << "circular buffer count        : " << g_circularbufCount << std::endl;
        std::cout << "circular buffer record count : " << g_circularbufSize << std::endl;
        std::cout << "File name                    : <context id>_" << g_fileName << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << std::endl;
    }

    return;
}

void
AtExitHandler()
{
    // Check for any error occured while pc sampling.
    CUptiResult cuptiStatus = cuptiGetLastError();
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        const char *pErrorString;
        cuptiGetResultString(cuptiStatus, &pErrorString);
        printf("%s: %d: error: function cuptiGetLastError() failed with error %s.\n", __FILE__, __LINE__, pErrorString);
        g_waitAtJoin = true;
        if (g_storeDataInFileThreadHandle.joinable())
        {
            g_storeDataInFileThreadHandle.join();
        }
        FreePreallocatedMemory();
        exit(EXIT_FAILURE);
    }

    if (g_running)
    {
        g_running = false;
        // Iterate over all context. If context is not destroyed then
        // disable PC sampling to flush remaining data to user's buffer.
        for (auto& itr: g_contextInfoMap)
        {
            auto GetPcSamplingData = [&](CUpti_PCSamplingGetDataParams &pcSamplingGetDataParams, ContextInfo *pContextInfo)
            {
                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, pContextInfo))
                {
                    printf("Error: NoFailed to get pc sampling data from Cupti\n");
                    g_waitAtJoin = true;
                    if (g_storeDataInFileThreadHandle.joinable())
                    {
                        g_storeDataInFileThreadHandle.join();
                    }
                    FreePreallocatedMemory();
                    exit(EXIT_FAILURE);
                }
            };

            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
            pcSamplingGetDataParams.ctx = itr.first;

            // For the case where hawdware buffer is full, remainingNumPc field from pcSamplingData will be 0.
            // Call GetPcSamplingDataFromCupti() function which calls cuptiPcSamplingGetData() API
            // which reports CUPTI_ERROR_OUT_OF_MEMORY for this case.
            if (itr.second->pcSamplingData.remainingNumPcs == 0)
            {
                GetPcSamplingData(pcSamplingGetDataParams, itr.second);
            }

            while (itr.second->pcSamplingData.remainingNumPcs > 0 || itr.second->pcSamplingData.totalNumPcs > 0)
            {
                GetPcSamplingData(pcSamplingGetDataParams, itr.second);
            }

            CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
            pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
            pcSamplingDisableParams.ctx = itr.first;
            CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

            if (!g_disableFileDump && itr.second->pcSamplingData.totalNumPcs > 0)
            {
                size_t remainingNumPcs = itr.second->pcSamplingData.remainingNumPcs;
                if (remainingNumPcs)
                {
                    std::cout << "WARNING : " << remainingNumPcs
                              << " records are discarded during cuptiPCSamplingDisable() since these can't be accommodated "
                              << "in the PC sampling buffer provided during the PC sampling configuration. Bigger buffer can mitigate this issue." << std::endl;
                }

                g_pcSampDataQueueMutex.lock();
                // It is quite possible that after pc sampling disabled cupti fill remaining records
                // collected lately from hardware in provided buffer during configuration.
                g_pcSampDataQueue.push(std::make_pair(&itr.second->pcSamplingData, itr.second));
                g_pcSampDataQueueMutex.unlock();
            }
        }

        if (g_buffersGetUtilisedFasterThanStore)
        {
            std::cout << "WARNING : Buffers get used faster than get stored in file. "
                      << "Suggestion is either increase size of buffer or increase number of buffers" << std::endl;
        }

        g_waitAtJoin = true;

        if (g_storeDataInFileThreadHandle.joinable())
        {
            g_storeDataInFileThreadHandle.join();
        }

        FreePreallocatedMemory();
    }

}

#ifdef _WIN32
typedef void (WINAPI* rtlExitUserProcess_t)(uint32_t exitCode);
rtlExitUserProcess_t Real_RtlExitUserProcess = NULL;

// Detour_RtlExitUserProcess
void WINAPI
Detour_RtlExitUserProcess(
    uint32_t exitCode)
{
    AtExitHandler();

    Real_RtlExitUserProcess(exitCode);
}
#endif

void RegisterAtExitHandler(void)
{
#ifdef _WIN32
    {
        // It's unsafe to use atexit(), static destructors, DllMain PROCESS_DETACH, etc.
        // because there's no way to guarantee the CUDA driver is still in a valid state
        // when you get to those, due to the undefined order of dynamic library tear-down
        // during process destruction.
        // Also, the first thing the Windows kernel does when any thread in a process
        // calls exit() is to immediately terminate all other threads, without any kind of
        // synchronization.
        // So the only valid time to do any in-process cleanup at exit() is before control
        // is passed to the kernel. Use Detours to intercept a low-level ntdll.dll function
        // "RtlExitUserProcess".
        int detourStatus = 0;
        FARPROC proc;

        // ntdll.dll will always be loaded, no need to load the library
        HMODULE ntDll = GetModuleHandle(TEXT("ntdll.dll"));
        if (!ntDll) {
            detourStatus = 1;
            goto DetourError;
        }

        proc = GetProcAddress(ntDll, "RtlExitUserProcess");
        if (!proc) {
            detourStatus = 1;
            goto DetourError;
        }
        Real_RtlExitUserProcess = (rtlExitUserProcess_t)proc;

        // Begin a detour transaction
        if (DetourTransactionBegin() != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        if (DetourUpdateThread(GetCurrentThread()) != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        DetourSetIgnoreTooSmall(TRUE);

        if (DetourAttach((void**)&Real_RtlExitUserProcess, (void*)Detour_RtlExitUserProcess) != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }

        // Commit the transaction
        if (DetourTransactionCommit() != ERROR_SUCCESS) {
            detourStatus = 1;
            goto DetourError;
        }
    DetourError:
        if (detourStatus != 0) {
            atexit(&AtExitHandler);
        }
    }
#else
    atexit(&AtExitHandler);
#endif
}

void CallbackHandler(
    void *pUserdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData)
{
    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {
            const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;

            switch (callbackId)
            {
                case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
                {
                    if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
                    {
                        std::map<CUcontext, ContextInfo*>::iterator contextStateMapItr = g_contextInfoMap.find(pCallbackInfo->context);
                        if (contextStateMapItr == g_contextInfoMap.end())
                        {
                            std::cout << "Error: Context not found in map." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        if (!contextStateMapItr->second->contextUid)
                        {
                            contextStateMapItr->second->contextUid = pCallbackInfo->contextUid;
                        }
                        // Get PC sampling data from cupti for each range. In such case records will get filled in provided buffer during configuration.
                        // It is recommend to collect those record using cuptiPCSamplingGetData() API.
                        // For _KERNEL_SERIALIZED mode each kernel data is one range.
                        if (g_pcSamplingCollectionMode == CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED)
                        {
                            // Collect all available records.
                            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                            pcSamplingGetDataParams.ctx = pCallbackInfo->context;

                            // Collect all records filled in provided buffer during configuration.
                            while (contextStateMapItr->second->pcSamplingData.totalNumPcs > 0)
                            {
                                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                                {
                                    printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                    exit(EXIT_FAILURE);
                                }
                            }
                            // Collect if any extra records which could not accommodated in provided buffer during configuration.
                            while (contextStateMapItr->second->pcSamplingData.remainingNumPcs > 0)
                            {
                                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                                {
                                    printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                    exit(EXIT_FAILURE);
                                }
                            }
                        }
                        else if (contextStateMapItr->second->pcSamplingData.remainingNumPcs >= g_circularbufSize)
                        {
                            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                            pcSamplingGetDataParams.ctx = pCallbackInfo->context;

                            if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                            {
                                printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                break;
            }
        }
        break;
        case CUPTI_CB_DOMAIN_RESOURCE:
        {
            const CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)pCallbackData;
            g_running = true;

            switch(callbackId)
            {
                case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
                {
                    {
                        if (g_verbose)
                        {
                            std::cout << "Injection - Context created" << std::endl;
                        }

                        // insert new entry for context.
                        ContextInfo *pContextInfo = (ContextInfo *)calloc(1, sizeof(ContextInfo));
                        MEMORY_ALLOCATION_CALL(pContextInfo);
                        g_contextInfoMutex.lock();
                        g_contextInfoMap.insert(std::make_pair(pResourceData->context, pContextInfo));
                        g_contextInfoMutex.unlock();

                        CUpti_PCSamplingEnableParams pcSamplingEnableParams = {};
                        pcSamplingEnableParams.size = CUpti_PCSamplingEnableParamsSize;
                        pcSamplingEnableParams.ctx = pResourceData->context;
                        CUPTI_API_CALL(cuptiPCSamplingEnable(&pcSamplingEnableParams));

                        ConfigureActivity(pResourceData->context);

                        g_circularBufferMutex.lock();
                        if (!g_allocatedCircularBuffers)
                        {
                            PreallocateBuffersForRecords();
                            g_allocatedCircularBuffers = true;
                        }
                        g_circularBufferMutex.unlock();
                    }
                }
                break;
                case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
                {
                    if (g_verbose)
                    {
                        std::cout << "Injection - Context destroy starting" << std::endl;
                    }
                    std::map<CUcontext, ContextInfo *>::iterator itr;
                    g_contextInfoMutex.lock();
                    itr = g_contextInfoMap.find(pResourceData->context);
                    if (itr == g_contextInfoMap.end())
                    {
                        std::cout << "Warning : This context not found in map of context which enabled PC sampling." << std::endl;
                    }
                    g_contextInfoMutex.unlock();

                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = itr->first;

                    // For the case where hawdware buffer is full, remainingNumPc field from pcSamplingData will be 0.
                    // Call GetPcSamplingDataFromCupti() function which calls cuptiPcSamplingGetData() API
                    // which reports CUPTI_ERROR_OUT_OF_MEMORY for this case.
                    if (itr->second->pcSamplingData.remainingNumPcs == 0)
                    {
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    while (itr->second->pcSamplingData.remainingNumPcs > 0 || itr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
                    pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
                    pcSamplingDisableParams.ctx = pResourceData->context;
                    CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

                    // It is quite possible that after pc sampling disabled cupti fill remaining records
                    // collected lately from hardware in provided buffer during configuration.
                    if (!g_disableFileDump && itr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        g_pcSampDataQueueMutex.lock();
                        g_pcSampDataQueue.push(std::make_pair(&itr->second->pcSamplingData, itr->second));
                        g_pcSampDataQueueMutex.unlock();
                    }

                    g_contextInfoMutex.lock();
                    g_contextInfoToFreeInEndVector.push_back(itr->second);
                    g_contextInfoMap.erase(itr);
                    g_contextInfoMutex.unlock();
                }
                break;
                case CUPTI_CBID_RESOURCE_MODULE_LOADED:
                {
                    g_contextInfoMutex.lock();
                    std::map<CUcontext, ContextInfo *>::iterator contextStateMapItr = g_contextInfoMap.find(pResourceData->context);
                    if (contextStateMapItr == g_contextInfoMap.end())
                    {
                        std::cout << "Error : Context not found in map" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    g_contextInfoMutex.unlock();
                    // Get PC sampling data from cupti for each range. In such case records will get filled in provided buffer during configuration.
                    // It is recommend to collect those record using cuptiPCSamplingGetData() API.
                    // If module get unloaded then afterwards records will belong to a new range.
                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = pResourceData->context;

                    // Collect all records filled in provided buffer during configuration.
                    while (contextStateMapItr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }
                    // Collect if any extra records which could not accommodated in provided buffer during configuration.
                    while (contextStateMapItr->second->pcSamplingData.remainingNumPcs > 0)
                    {
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                break;
            }
        }
        break;
        default :
            break;
    }
}

#ifdef _WIN32
extern "C" __declspec(dllexport) int InitializeInjection(void)
#else
extern "C" int InitializeInjection(void)
#endif
{
    g_initializeInjectionMutex.lock();
    if (!g_initializedInjection)
    {
        std::cout << "... Initialize injection ..." << std::endl;

        ReadInputParams();

        CUpti_SubscriberHandle subscriber;
        CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)&CallbackHandler, NULL));

        // Subscribe for all the launch callbacks.
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice));
        // Subscribe for module and context callbacks.
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));

        g_initializedInjection = true;
    }

    RegisterAtExitHandler();
    g_initializeInjectionMutex.unlock();

    return 1;
}
