// Copyright 2020-2022 NVIDIA Corporation. All rights reserved
//
// The sample provides the generic workflow for querying various properties of metrics which are available as part of
// the Profiling APIs. In this particular case we are querying for number of passes and collection method for a list of metrics.
//
// Number of passes : It gives the number of passes required for collection of the metric as some of the metric
// cannot be collected in single pass due to hardware or software limitation, we need to replay the exact same
// set of GPU workloads multiple times.
//
// Collection method : It gives the source of the metric (HW or SW). Most of metric are provided by hardware but for
// some metric we have to instrument the kernel to collect the metric. Further these metrics cannot be combined with
// any other metrics in the same pass as otherwise instrumented code will also contribute to the metric value.
//

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

// CUPTI headers
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include "helper_cupti.h"

// NVPW headers
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>

#include <Parser.h>
#include <Utils.h>
#include <List.h>

#define FORMAT_METRIC_DETAILS(stream, metricName, numOfPasses, collectionMethod, isCSVformat)           \
    if(isCSVformat) {                                                                                   \
        stream  << metricName  << ","                                                                   \
                << numOfPasses << ","                                                                   \
                << collectionMethod << "\n";                                                            \
    } else {                                                                                            \
        stream  << std::setw(80) << std::left << metricName  << "\t"                                    \
                << std::setw(15) << std::left << numOfPasses << "\t"                                    \
                << std::setw(15) << std::left << collectionMethod << "\n";                              \
    }

#define PRINT_METRIC_DETAILS(stream, outputStream, isCSVformat)                                         \
{                                                                                                       \
    FORMAT_METRIC_DETAILS(stream, "Metric Name", "Num of Passes", "Collection Method", isCSVformat)     \
    std::string metricName, numOfPasses, collectionMethod;                                              \
    while (outputStream >> metricName >> numOfPasses >> collectionMethod) {                             \
        FORMAT_METRIC_DETAILS(stream, metricName, numOfPasses, collectionMethod, isCSVformat)           \
    }                                                                                                   \
}

std::string
GetMetricCollectionMethod(
    std::string metricName)
{
    const std::string SW_CHECK = "sass";
    if (metricName.find(SW_CHECK) != std::string::npos)
    {
        return "SW";
    }

    return "HW";
}

bool
GetRawMetricRequests(
    std::string chipName,
    std::string metricName,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    const uint8_t *pCounterAvailabilityImage)
{
    std::string reqName;
    bool isolated = true;
    bool keepInstances = true;

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParam.pChipName = chipName.c_str();
    calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

    std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
    metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
    metricEvaluatorInitializeParams.pChipName = chipName.c_str();
    metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
    NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

    std::vector<const char*> rawMetricNames;
    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    keepInstances = true;

    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

    std::vector<const char*> rawDependencies;
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDependenciesParms = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
    getMetricRawDependenciesParms.pMetricsEvaluator = metricEvaluator;
    getMetricRawDependenciesParms.pMetricEvalRequests = &metricEvalRequest;
    getMetricRawDependenciesParms.numMetricEvalRequests = 1;
    getMetricRawDependenciesParms.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    getMetricRawDependenciesParms.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));
    rawDependencies.resize(getMetricRawDependenciesParms.numRawDependencies);
    getMetricRawDependenciesParms.ppRawDependencies = rawDependencies.data();
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));

    for (size_t i = 0; i < rawDependencies.size(); ++i)
    {
        rawMetricNames.push_back(rawDependencies[i]);
    }

    for (auto& rawMetricName : rawMetricNames)
    {
        NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
        metricRequest.pMetricName = rawMetricName;
        metricRequest.isolated = isolated;
        metricRequest.keepInstances = keepInstances;
        rawMetricRequests.push_back(metricRequest);
    }

    NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
    metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));

    return true;
}


bool
GetMetricDetails(
    std::string metricName,
    std::string chipName,
    std::stringstream& outputStream,
    const uint8_t *pCounterAvailabilityImage)
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    if (!GetRawMetricRequests(chipName, metricName, rawMetricRequests, pCounterAvailabilityImage))
    {
        printf("Error: Failed to get raw metrics.\n");

        return false;
    }

    NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
    rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    rawMetricsConfigCreateParams.pChipName = chipName.c_str();
    rawMetricsConfigCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams));

    NVPA_RawMetricsConfig *pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

    NVPW_RawMetricsConfig_IsAddMetricsPossible_Params isAddMetricsPossibleParams = { NVPW_RawMetricsConfig_IsAddMetricsPossible_Params_STRUCT_SIZE };
    isAddMetricsPossibleParams.pRawMetricsConfig = pRawMetricsConfig;
    isAddMetricsPossibleParams.pRawMetricRequests = rawMetricRequests.data();
    isAddMetricsPossibleParams.numMetricRequests = rawMetricRequests.size();
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_IsAddMetricsPossible(&isAddMetricsPossibleParams));

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
    addMetricsParams.numMetricRequests = rawMetricRequests.size();
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = { NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE };
    rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = pRawMetricsConfig;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetNumPasses(&rawMetricsConfigGetNumPassesParams));

    // No Nesting of ranges in case of CUPTI_AutoRange, in AutoRange
    // the range is already at finest granularity of every kernel Launch so numNestingLevels = 1
    size_t numNestingLevels = 1;
    size_t numIsolatedPasses = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    size_t numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;
    size_t numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;
    std::string collectionMethod = GetMetricCollectionMethod(metricName);

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params*)&rawMetricsConfigDestroyParams));

    outputStream << metricName << " " << numOfPasses << " " << collectionMethod << "\n";

    return true;
}

int
main(
    int argc,
    char *argv[])
{
    std::vector<std::string> metricNames;
    int deviceCount;

    int deviceNum = 0;
    std::string chipName;
    bool bIsCSVformat = false;
    char* metricName;
    std::string exportFileName;

    for (int i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {
            printf("Usage: %s --device [device_num] --chip [chip name] --metrics [metric_names comma separated] --csv --file [filename]\n", argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (strcmp(arg, "--device") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add device number for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            deviceNum = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(arg, "--chip") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add chip name for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            chipName = argv[i + 1];
            i++;
        }
        else if (strcmp(arg, "--metrics") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add metric names for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            metricName = strtok(argv[i+1], ",");
            while (metricName != NULL)
            {
                metricNames.push_back(metricName);
                metricName = strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--csv") == 0)
        {
            bIsCSVformat = true;
        }
        else if (strcmp(arg, "--file") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add file name for exporting metric details.\n");
                exit(EXIT_FAILURE);
            }
            exportFileName = argv[i + 1];
            i++;
        }
        else
        {
            printf("Error!! Invalid Arguments\n");
            printf("Usage: %s --device [device_num] --chip [chip name] --metrics [metric_names comma separated] --csv --file [filename]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    std::vector<uint8_t> counterAvailabilityImage;
    if (chipName.empty())
    {
        DRIVER_API_CALL(cuInit(0));
        DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

        if (deviceCount == 0)
        {
            printf("There is no device supporting CUDA.\n");
            exit(EXIT_WAIVED);
        }
        printf("CUDA Device Number: %d\n", deviceNum);

        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

        int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
        DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
        DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
        printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

        // Initialize profiler API and test device compatibility
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
        CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
        params.cuDevice = deviceNum;
        CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

        if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
        {
            ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;

            if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
            }

            if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
            }

            if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
            }
            else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
            {
                ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
            }

            if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
            }

            if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
            }

            if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            {
                ::std::cerr << "\tWSL is not supported" << ::std::endl;
            }
            exit(EXIT_WAIVED);
        }

        /* Get chip name for the cuda  device */
        CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
        getChipNameParams.deviceIndex = deviceNum;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        chipName = getChipNameParams.pChipName;

        CUcontext cuContext;
        DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

        CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
        getCounterAvailabilityParams.ctx = cuContext;
        CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

        counterAvailabilityImage.clear();
        counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
        getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
        CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
    }
    printf("Queried Chip : %s \n", chipName.c_str());

    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    RETURN_IF_NVPW_ERROR(false, NVPW_InitializeHost(&initializeHostParams));

    std::stringstream outputStream;
    if (metricNames.empty())
    {
        auto listSubMetrics = true;
        std::vector<std::string> metricsList;
        if (NV::Metric::Enum::ExportSupportedMetrics(chipName.c_str(), listSubMetrics, counterAvailabilityImage.data(), metricsList))
        {
            std::cout << "Total metrics on the chip " << metricsList.size() << "\n";
            for (size_t i = 0; i < metricsList.size(); i++)
            {
                if (!GetMetricDetails(metricsList[i], chipName, outputStream, counterAvailabilityImage.data()))
                {
                    printf("Error!! Failed to get the metric details\n");
                    exit(EXIT_WAIVED);
                }
            }
        }
    }
    else
    {
        for (auto metricName : metricNames)
        {
            if (!GetMetricDetails(metricName, chipName, outputStream, counterAvailabilityImage.data()))
            {
                printf("Error!! Failed to get the metric details\n");
                exit(EXIT_WAIVED);
            }
        }
    }

    if (exportFileName.empty())
    {
       PRINT_METRIC_DETAILS(std::cout, outputStream, bIsCSVformat);
    }
    else
    {
        std::ofstream outputFile(exportFileName);
        if (outputFile.is_open())
        {
            PRINT_METRIC_DETAILS(outputFile, outputStream, bIsCSVformat);
            outputFile.close();
            printf("Metric details has been written to %s file.\n", exportFileName.c_str());
        }
    }

    exit(EXIT_SUCCESS);
}
