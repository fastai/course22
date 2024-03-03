#if !defined(_PC_SAMPLING_UTILITY_HELPER_H_)
#define _PC_SAMPLING_UTILITY_HELPER_H_

// System headers
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <string.h>

// CUPTI headers
#include <cupti_pcsampling_util.h>
#include <cupti_pcsampling.h>
#include "helper_cupti.h"

using namespace CUPTI::PcSamplingUtil;

typedef struct ModuleDetails_st
{
    uint32_t cubinSize;
    void *pCubinImage;
} ModuleDetails;

std::string fileName;
PcSamplingStallReasons pcSamplingStallReasonsRetrieve;
std::vector<CUpti_PCSamplingData> buffersRetrievedDataVector;
std::map<uint64_t, ModuleDetails> crcModuleMap;
CUpti_PCSamplingCollectionMode collectionMode;

bool disableMerge;
bool disablePcInfoPrints;
bool disableSourceCorrelation;
bool verbose;

static void
Init()
{
    fileName = "";
    pcSamplingStallReasonsRetrieve = {};
    collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;

    disableMerge = false;
    disablePcInfoPrints = false;
    disableSourceCorrelation = false;
    verbose = false;
}

static void
PrintUsage()
{
    printf("Usage: pc_sampling_utility\n");
    printf("       --help                            : Displays help message.\n");
    printf("       --file-name                       : Name of the file to parse and print data.\n");
    printf("       --disable-merge                   : Disable merge of buffers.\n");
    printf("       --disable-pc-info-prints          : Disable PC records info prints.\n");
    printf("       --disable-source-correlation      : Disable Source correlation.\n");
    printf("       --verbose                         : Enable verbose prints.\n");

    exit(EXIT_SUCCESS);
}

static void
ParseCommandLineArgs(
    int argc,
    char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Pass file to parse." << std::endl;
        PrintUsage();
    }

    for (int i = 1; i < argc; i++)
    {
        if ((stricmp(argv[i], "--help") == 0) ||
            (stricmp(argv[i], "-help") == 0))
        {
            PrintUsage();
        }
        else if ((stricmp(argv[i], "--file-name") == 0) ||
                (stricmp(argv[i], "-file-name") == 0))
        {
            if (argc < i + 2)
            {
                std::cout << "ERROR : Pass file to parse." << std::endl;
                PrintUsage();
            }
            fileName = argv[i+1];
            i++;
        }
        else if ((stricmp(argv[i], "--disable-merge") == 0) ||
                (stricmp(argv[i], "-disable-merge") == 0))
        {
            disableMerge = true;
        }
        else if ((stricmp(argv[i], "--disable-pc-info-prints") == 0) ||
                (stricmp(argv[i], "-disable-pc-info-prints") == 0))
        {
            disablePcInfoPrints = true;
        }
        else if ((stricmp(argv[i], "--disable-source-correlation") == 0) ||
                (stricmp(argv[i], "-disable-source-correlation") == 0))
        {
            disableSourceCorrelation = true;
        }
        else if ((stricmp(argv[i], "--verbose") == 0) ||
                (stricmp(argv[i], "-verbose") == 0))
        {
            verbose = true;
        }
        else
        {
            std::cout << "Unknown option: " << argv[i] << std::endl;
            PrintUsage();
        }

    }
}

/**
 * Function Info :
 * Store stall reasons as per vector index for ease of access
 */
static std::string
GetStallReason(
    uint32_t pcSamplingStallReasonIndex)
{
    for (size_t i = 0; i < pcSamplingStallReasonsRetrieve.numStallReasons; i++)
    {
        if (pcSamplingStallReasonsRetrieve.stallReasonIndex[i] == pcSamplingStallReasonIndex)
        {
            return pcSamplingStallReasonsRetrieve.stallReasons[i];
        }
    }

    return "ERROR_STALL_REASON_INDEX_NOT_FOUND";
}

static void
PrintConfigurationDetails(
    CUptiUtil_GetPcSampDataParams &getPcSampDataParams)
{
    std::cout << "========================== Configuration info ==========================" << std::endl;

    for (size_t i=0; i<getPcSampDataParams.numAttributes; i++)
    {
        switch (getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeType)
        {
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD:
                std::cout << "sampling period: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.samplingPeriodData.samplingPeriod << std::endl;
                break;
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON:
            {
                std::cout << "selected stall reasons count: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.stallReasonData.stallReasonCount << std::endl;
                std::cout << "selected stall reasons: " << std::endl;
                for (size_t stallReasonIndex=0; stallReasonIndex < getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.stallReasonData.stallReasonCount; stallReasonIndex++)
                {
                    std::cout << GetStallReason(getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.stallReasonData.pStallReasonIndex[stallReasonIndex]) << ", ";
                    if ((stallReasonIndex+1) % 5 == 0)
                        std::cout << std::endl;
                }
                std::cout << std::endl;
                break;
            }
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE:
                std::cout << "scratch buffer size: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.scratchBufferSizeData.scratchBufferSize << std::endl;
                break;
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE:
                std::cout << "hw buffer size: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.hardwareBufferSizeData.hardwareBufferSize << std::endl;
                break;
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE:
                std::cout << "collection mode: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.collectionModeData.collectionMode << std::endl;
                break;
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL:
                std::cout << "enable start stop: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.enableStartStopControlData.enableStartStopControl << std::endl;
                break;
            case CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT:
                std::cout << "output data format: " << getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.outputDataFormatData.outputDataFormat << std::endl;
                break;
            default:
                break;
        }
    }

    std::cout << "========================================================================" << std::endl;
}

static void
PrintRetrievedPcSampData()
{
    for (size_t pcSampBufferIndex = 0; pcSampBufferIndex < buffersRetrievedDataVector.size(); pcSampBufferIndex++)
    {
        std::cout << "========================== PC Records Buffer Info ==========================" << std::endl;
        std::cout << "Buffer Number: " << pcSampBufferIndex + 1
                  << ", Range Id: " << buffersRetrievedDataVector[pcSampBufferIndex].rangeId
                  << ", Count of PC records: " << buffersRetrievedDataVector[pcSampBufferIndex].totalNumPcs
                  << ", Total Samples: " << buffersRetrievedDataVector[pcSampBufferIndex].totalSamples
                  << ", Total Dropped Samples: " << buffersRetrievedDataVector[pcSampBufferIndex].droppedSamples;

        if (CHECK_PC_SAMPLING_STRUCT_FIELD_EXISTS(CUpti_PCSamplingData, nonUsrKernelsTotalSamples, buffersRetrievedDataVector[pcSampBufferIndex].size))
        {
            std::cout << ", Non User Kernels Total Samples: " << buffersRetrievedDataVector[pcSampBufferIndex].nonUsrKernelsTotalSamples;
        }
        std::cout << std::endl;

        for(size_t i=0 ; i < buffersRetrievedDataVector[pcSampBufferIndex].totalNumPcs; i++)
        {
            std::cout << ", cubinCrc: " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].cubinCrc
                      << ", functionName: " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].functionName
                      << ", functionIndex: " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].functionIndex
                      << ", pcOffset: " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].pcOffset
                      << ", stallReasonCount: " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].stallReasonCount;

            for (size_t k=0; k < buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].stallReasonCount; k++)
            {
                std::cout << ", " << GetStallReason(buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].stallReason[k].pcSamplingStallReasonIndex)
                          << ": " << buffersRetrievedDataVector[pcSampBufferIndex].pPcData[i].stallReason[k].samples;
            }
            std::cout << std::endl;
        }
    }
}

/**
 * Function Info :
 * Open file
 * Get header info using CuptiUtilGetHeaderData()  CUPTI UTIL API
 * Iterate over available number of buffers.
 *    Read buffer info using CuptiUtilGetBufferInfo() CUPTI UTIL API
 *    Allocate memory for PC samp data buffers
 *    Retrieve PC samp data using CuptiUtilGetPcSampData() CUPTI UTIL API
 */
static void
RetrievePcSampData()
{
    std::ifstream fileHandler(fileName, std::ios::out | std::ios::binary);

    if (!fileHandler)
    {
        std::cerr << "Cannot open file : " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    CUptiUtil_GetHeaderDataParams getHeaderDataParams = {};
    getHeaderDataParams.size = CUptiUtil_GetHeaderDataParamsSize;
    getHeaderDataParams.fileHandler = &fileHandler;

    CUPTI_UTIL_CALL(CuptiUtilGetHeaderData(&getHeaderDataParams));

    if (verbose)
    {
        std::cout << "Total buffers available in file " << fileName << ": " << getHeaderDataParams.headerInfo.totalBuffers << std::endl;
    }

    for (size_t i = 0; i < getHeaderDataParams.headerInfo.totalBuffers; i++)
    {
        CUptiUtil_GetBufferInfoParams getBufferInfoParams = {};
        getBufferInfoParams.size = CUptiUtil_GetBufferInfoParamsSize;
        getBufferInfoParams.fileHandler = &fileHandler;

        CUPTI_UTIL_CALL(CuptiUtilGetBufferInfo(&getBufferInfoParams));

        CUpti_PCSamplingData buffersRereivedData = {0};
        buffersRereivedData.pPcData = (CUpti_PCSamplingPCData *) calloc (getBufferInfoParams.bufferInfoData.recordCount, sizeof(CUpti_PCSamplingPCData));
        MEMORY_ALLOCATION_CALL(buffersRereivedData.pPcData);
        for (size_t j=0; j<getBufferInfoParams.bufferInfoData.recordCount; j++)
        {
            buffersRereivedData.pPcData[j].stallReason = (CUpti_PCSamplingStallReason *)calloc(getBufferInfoParams.bufferInfoData.numSelectedStallReasons, sizeof(CUpti_PCSamplingStallReason));
        }

        if (i == 0)
        {
            char **pStallReasonsRetrieve = (char **)calloc(getBufferInfoParams.bufferInfoData.numStallReasons, sizeof(char*));
            MEMORY_ALLOCATION_CALL(pStallReasonsRetrieve);
            for (size_t i = 0; i < getBufferInfoParams.bufferInfoData.numStallReasons; i++)
            {
                pStallReasonsRetrieve[i] = (char *)calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char));
                MEMORY_ALLOCATION_CALL(pStallReasonsRetrieve[i]);
            }
            uint32_t *pStallReasonIndexRetrieve = (uint32_t *)calloc(getBufferInfoParams.bufferInfoData.numStallReasons, sizeof(uint32_t));
            MEMORY_ALLOCATION_CALL(pStallReasonIndexRetrieve);

            pcSamplingStallReasonsRetrieve.numStallReasons = getBufferInfoParams.bufferInfoData.numStallReasons;
            pcSamplingStallReasonsRetrieve.stallReasonIndex = pStallReasonIndexRetrieve;
            pcSamplingStallReasonsRetrieve.stallReasons = pStallReasonsRetrieve;

            CUpti_PCSamplingConfigurationInfo getSampPeriod = {};
            getSampPeriod.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;

            CUpti_PCSamplingConfigurationInfo getStallReason = {};
            uint32_t *pGetStallReasonIndex = (uint32_t *)calloc(getBufferInfoParams.bufferInfoData.numSelectedStallReasons, sizeof(uint32_t));
            MEMORY_ALLOCATION_CALL(pGetStallReasonIndex);
            getStallReason.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
            getStallReason.attributeData.stallReasonData.pStallReasonIndex = pGetStallReasonIndex;

            CUpti_PCSamplingConfigurationInfo getScratchBufferSize = {};
            getScratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;

            CUpti_PCSamplingConfigurationInfo getHwBufferSize = {};
            getHwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;

            CUpti_PCSamplingConfigurationInfo getCollectionMode = {};
            getCollectionMode.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;

            CUpti_PCSamplingConfigurationInfo getEnableStartStop = {};
            getEnableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;

            CUpti_PCSamplingConfigurationInfo getOutputDataFormat = {};
            getOutputDataFormat.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;

            std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfoRetrieve;
            pcSamplingConfigurationInfoRetrieve.push_back(getSampPeriod);
            pcSamplingConfigurationInfoRetrieve.push_back(getStallReason);
            pcSamplingConfigurationInfoRetrieve.push_back(getScratchBufferSize);
            pcSamplingConfigurationInfoRetrieve.push_back(getHwBufferSize);
            pcSamplingConfigurationInfoRetrieve.push_back(getCollectionMode);
            pcSamplingConfigurationInfoRetrieve.push_back(getEnableStartStop);
            pcSamplingConfigurationInfoRetrieve.push_back(getOutputDataFormat);

            CUptiUtil_GetPcSampDataParams getPcSampDataParams = {};
            getPcSampDataParams.size = CUptiUtil_GetPcSampDataParamsSize;
            getPcSampDataParams.fileHandler = &fileHandler;
            getPcSampDataParams.bufferType = PC_SAMPLING_BUFFER_PC_TO_COUNTER_DATA;
            getPcSampDataParams.pBufferInfoData = &getBufferInfoParams.bufferInfoData;
            getPcSampDataParams.pSamplingData = (void*)&buffersRereivedData;
            getPcSampDataParams.numAttributes = pcSamplingConfigurationInfoRetrieve.size();
            getPcSampDataParams.pPCSamplingConfigurationInfo =  pcSamplingConfigurationInfoRetrieve.data();
            getPcSampDataParams.pPcSamplingStallReasons = &pcSamplingStallReasonsRetrieve;

            CUPTI_UTIL_CALL(CuptiUtilGetPcSampData(&getPcSampDataParams));

            for (size_t i = 0; i < getPcSampDataParams.numAttributes; i++)
            {
                if (getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeType == CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE)
                {
                    collectionMode = getPcSampDataParams.pPCSamplingConfigurationInfo[i].attributeData.collectionModeData.collectionMode;
                    break;
                }
            }

            if (verbose)
            {
                PrintConfigurationDetails(getPcSampDataParams);
            }
        }
        else
        {
            CUptiUtil_GetPcSampDataParams pGetOnlyPcSampDataParams = {};
            pGetOnlyPcSampDataParams.size = CUptiUtil_GetPcSampDataParamsSize;
            pGetOnlyPcSampDataParams.fileHandler = &fileHandler;
            pGetOnlyPcSampDataParams.bufferType = PC_SAMPLING_BUFFER_PC_TO_COUNTER_DATA;
            pGetOnlyPcSampDataParams.pBufferInfoData = &getBufferInfoParams.bufferInfoData;
            pGetOnlyPcSampDataParams.pSamplingData = (void*)&buffersRereivedData;
            pGetOnlyPcSampDataParams.numAttributes = 0;
            pGetOnlyPcSampDataParams.pPCSamplingConfigurationInfo =  NULL;
            pGetOnlyPcSampDataParams.pPcSamplingStallReasons = NULL;

            CUPTI_UTIL_CALL(CuptiUtilGetPcSampData(&pGetOnlyPcSampDataParams));
        }

        buffersRetrievedDataVector.push_back(buffersRereivedData);
    }

    fileHandler.close();
}

/**
 * Function Info :
 * read file size
 * read file
 * compute hash on module using cuptiGetCubinCrc() CUPTI API.
 * and store it in map for every Cubin.
 */
static void
FillCrcModuleMap()
{
    for(int i = 1 ;; i++)
    {
        ModuleDetails moduleDetailsStruct = {};
        std::string cubinFileName = std::to_string(i) + ".cubin";

        std::ifstream fileHandler(cubinFileName, std::ios::binary | std::ios::ate);

        if (!fileHandler)
        {
            break;
        }

        moduleDetailsStruct.cubinSize = fileHandler.tellg();

        if (!fileHandler.seekg(0, std::ios::beg))
        {
            std::cerr << "Unable to find size for cubin file " << cubinFileName << std::endl;
            exit(EXIT_FAILURE);
        }

        moduleDetailsStruct.pCubinImage = malloc(sizeof(char) * moduleDetailsStruct.cubinSize);
        MEMORY_ALLOCATION_CALL(moduleDetailsStruct.pCubinImage);

        fileHandler.read((char*)moduleDetailsStruct.pCubinImage, moduleDetailsStruct.cubinSize);

        fileHandler.close();

        if (verbose)
        {
            std::cout << "Read cubin file " << cubinFileName << std::endl;
        }

        // Find cubin CRC
        CUpti_GetCubinCrcParams cubinCrcParams = {0};
        cubinCrcParams.size = CUpti_GetCubinCrcParamsSize;
        cubinCrcParams.cubinSize = moduleDetailsStruct.cubinSize;
        cubinCrcParams.cubin = moduleDetailsStruct.pCubinImage;

        CUPTI_API_CALL(cuptiGetCubinCrc(&cubinCrcParams));

        uint64_t cubinCrc = cubinCrcParams.cubinCrc;
        crcModuleMap.insert(std::make_pair(cubinCrc, moduleDetailsStruct));
    }

    if (verbose)
    {
        std::cout << std::endl;
    }
}

/**
 * Function Info :
 * Merge all retrieved buffer using CuptiUtilMergePcSampData() CUPTI UTIL API.
 */
static void
MergePcSampDataBuffers(
    CUpti_PCSamplingData **mergedPcSampDataBuffer,
    size_t& numMergedPcSampDataBuffer)
{
    CUptiUtil_MergePcSampDataParams mergePcSampDataParams = {};
    mergePcSampDataParams.size = CUptiUtil_MergePcSampDataParamsSize;
    mergePcSampDataParams.numberOfBuffers = buffersRetrievedDataVector.size();
    mergePcSampDataParams.PcSampDataBuffer = buffersRetrievedDataVector.data();
    mergePcSampDataParams.MergedPcSampDataBuffers = mergedPcSampDataBuffer;
    mergePcSampDataParams.numMergedBuffer = &numMergedPcSampDataBuffer;

    CUPTI_UTIL_CALL(CuptiUtilMergePcSampData(&mergePcSampDataParams));

    if (verbose)
    {
        std::cout << buffersRetrievedDataVector.size() <<" buffers merged into " << numMergedPcSampDataBuffer << " buffer/s." << std::endl;
    }
}

/**
 * Function Info :
 * Iterate over all PC samp data buffers
 *     Iterate over each PC record
 *         Find Cubin in which PC belongs using cubin crc.
 *         Do source correlation using cuptiGetSassToSourceCorrelation() CUPTI API.
 */
static void
SourceCorrelation(
    CUpti_PCSamplingData *pPcSampDataBuffer,
    size_t numPcSampDataBuffer)
{
    std::map<uint64_t, ModuleDetails>::iterator itr;
    size_t numPcNoCubin = 0;
    size_t numPcNoLineinfo = 0;

    for (size_t pcSampBufferIndex = 0; pcSampBufferIndex < numPcSampDataBuffer; pcSampBufferIndex++)
    {
        std::cout << "========================== PC Records Buffer Info ==========================" << std::endl;
        std::cout << "Buffer Number: " << pcSampBufferIndex + 1
                  << ", Range Id: " << pPcSampDataBuffer[pcSampBufferIndex].rangeId
                  << ", Count of PC records: " << pPcSampDataBuffer[pcSampBufferIndex].totalNumPcs
                  << ", Total Samples: " << pPcSampDataBuffer[pcSampBufferIndex].totalSamples
                  << ", Total Dropped Samples: " << pPcSampDataBuffer[pcSampBufferIndex].droppedSamples;
        if (CHECK_PC_SAMPLING_STRUCT_FIELD_EXISTS(CUpti_PCSamplingData, nonUsrKernelsTotalSamples, buffersRetrievedDataVector[pcSampBufferIndex].size))
        {
            std::cout << ", Non User Kernels Total Samples: " << buffersRetrievedDataVector[pcSampBufferIndex].nonUsrKernelsTotalSamples;
        }
        std::cout << std::endl;



        for(size_t i = 0 ; i < pPcSampDataBuffer[pcSampBufferIndex].totalNumPcs; i++)
        {
            // find matching cubinCrc entry in map
            itr = crcModuleMap.find(pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].cubinCrc);

            if (itr == crcModuleMap.end())
            {
                numPcNoCubin++;

                if (!disablePcInfoPrints)
                {
                    std::cout << "functionName: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].functionName
                              << ", functionIndex: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].functionIndex
                              << ", pcOffset: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].pcOffset
                              << ", lineNumber:0"
                              << ", fileName: " << "ERROR_NO_CUBIN"
                              << ", dirName: "
                              << ", stallReasonCount: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReasonCount;

                    for (size_t k=0; k < pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReasonCount; k++)
                    {
                        std::cout << ", " << GetStallReason(pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReason[k].pcSamplingStallReasonIndex)
                                  << ": " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReason[k].samples;
                    }
                    std::cout << std::endl;
                }

                continue;
            }

            if (!disablePcInfoPrints)
            {
                std::cout << "functionName: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].functionName
                          << ", functionIndex: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].functionIndex
                          << ", pcOffset: " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].pcOffset;
            }

            CUpti_GetSassToSourceCorrelationParams pCSamplingGetSassToSourceCorrelationParams = {0};
            pCSamplingGetSassToSourceCorrelationParams.size = CUpti_GetSassToSourceCorrelationParamsSize;
            pCSamplingGetSassToSourceCorrelationParams.functionName = pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].functionName;
            pCSamplingGetSassToSourceCorrelationParams.pcOffset = pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].pcOffset;
            pCSamplingGetSassToSourceCorrelationParams.cubin = itr->second.pCubinImage;
            pCSamplingGetSassToSourceCorrelationParams.cubinSize = itr->second.cubinSize;

            CUptiResult cuptiResult = cuptiGetSassToSourceCorrelation(&pCSamplingGetSassToSourceCorrelationParams);

            if (!disablePcInfoPrints)
            {
                if (cuptiResult == CUPTI_SUCCESS)
                {
                    std::cout << ", lineNumber: " << pCSamplingGetSassToSourceCorrelationParams.lineNumber
                              << ", fileName: " << pCSamplingGetSassToSourceCorrelationParams.fileName
                              << ", dirName: " << pCSamplingGetSassToSourceCorrelationParams.dirName;

                    free(pCSamplingGetSassToSourceCorrelationParams.fileName);
                    free(pCSamplingGetSassToSourceCorrelationParams.dirName);
                }
                else
                {
                    // It is possible that extracted cubins does not have lineinfo.
                    // It is recommended to build application/libraries with nvcc option lineinfo.
                    numPcNoLineinfo++;
                    std::cout << ", lineNumber: 0"
                              << ", fileName: " << "ERROR_NO_LINEINFO"
                              << ", dirName: ";
                }

                std::cout << ", stallReasonCount: " <<pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReasonCount;

                for (size_t k=0; k < pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReasonCount; k++)
                {
                    std::cout << ", " << GetStallReason(pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReason[k].pcSamplingStallReasonIndex)
                              << ": " << pPcSampDataBuffer[pcSampBufferIndex].pPcData[i].stallReason[k].samples;
                }
                std::cout << std::endl;
            }
        }
    }

    if (numPcNoCubin)
    {
        std::cerr << std::endl << "WARNING :: For these many PCs did not find cubin of same CRC: " << numPcNoCubin << std::endl;
    }
    if (numPcNoLineinfo)
    {
        std::cerr << std::endl << "WARNING :: These many PCs belongs to cubin which don't have lineinfo: " << numPcNoLineinfo << std::endl;
    }
}

static void
FreePcSampStallReasonsMemory()
{
    for (size_t i = 0; i < pcSamplingStallReasonsRetrieve.numStallReasons; i++)
    {
        free(pcSamplingStallReasonsRetrieve.stallReasons[i]);
    }
    free(pcSamplingStallReasonsRetrieve.stallReasons);
    free(pcSamplingStallReasonsRetrieve.stallReasonIndex);
}

static void
FreePcSampDataBuffers(CUpti_PCSamplingData *pcSampData, size_t numBuffers)
{
    for (size_t i=0; i<numBuffers; i++)
    {
        for (size_t j=0; j<pcSampData[i].totalNumPcs; j++)
        {
            free(pcSampData[i].pPcData[j].stallReason);
            free(pcSampData[i].pPcData[j].functionName);
        }
        free(pcSampData[i].pPcData);
    }
}

static void
FreeCrcModuleMapMemory()
{
    for (auto itr = crcModuleMap.begin(); itr != crcModuleMap.end(); itr++)
    {
        free(itr->second.pCubinImage);
    }
}

#endif