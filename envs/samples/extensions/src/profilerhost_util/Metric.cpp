#include <Metric.h>
#include <Parser.h>
#include <Utils.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <iostream>
#include <ScopeExit.h>

namespace NV {
    namespace Metric {
        namespace Config {

            bool GetRawMetricRequests(std::string chipName,
                                      const std::vector<std::string>& metricNames,
                                      std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                                      const uint8_t* pCounterAvailabilityImage)
            {
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

                bool isolated = true;
                bool keepInstances = true;
                std::vector<const char*> rawMetricNames;
                for (auto& metricName : metricNames)
                {
                    std::string reqName;
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

            bool GetConfigImage(std::string chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& configImage, const uint8_t* pCounterAvailabilityImage)
            {
                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                GetRawMetricRequests(chipName, metricNames, rawMetricRequests, pCounterAvailabilityImage);

                NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
                rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
                rawMetricsConfigCreateParams.pChipName = chipName.c_str();
                rawMetricsConfigCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams));
                NVPA_RawMetricsConfig* pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;

                if(pCounterAvailabilityImage)
                {
                    NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
                    setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
                    setCounterAvailabilityParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams));
                }

                NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
                rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
                SCOPE_EXIT([&]() { NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams); });

                NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
                beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

                NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
                addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

                NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
                endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

                NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
                generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

                NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
                getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                getConfigImageParams.bytesAllocated = 0;
                getConfigImageParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                configImage.resize(getConfigImageParams.bytesCopied);
                getConfigImageParams.bytesAllocated = configImage.size();
                getConfigImageParams.pBuffer = configImage.data();
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                return true;
            }

            bool GetCounterDataPrefixImage(std::string chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& counterDataImagePrefix, const uint8_t* pCounterAvailabilityImage)
            {
                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                GetRawMetricRequests(chipName, metricNames, rawMetricRequests, pCounterAvailabilityImage);

                NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE };
                counterDataBuilderCreateParams.pChipName = chipName.c_str();
                counterDataBuilderCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

                NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
                counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                SCOPE_EXIT([&]() { NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams); });

                NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

                size_t counterDataPrefixSize = 0;
                NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
                getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                getCounterDataPrefixParams.bytesAllocated = 0;
                getCounterDataPrefixParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);
                getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
                getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                return true;
            }
        
        }
    }
}
