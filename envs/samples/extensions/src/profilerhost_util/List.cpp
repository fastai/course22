#include <List.h>
#include <Utils.h>
#include <iostream>
#include <vector>
#include <string>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <ScopeExit.h>

namespace NV {
    namespace Metric {
        namespace Enum {

            std::string GetMetricRollupOpString(NVPW_RollupOp rollupOp)
            {
                switch (rollupOp)
                {
                    case NVPW_ROLLUP_OP_AVG:
                        return ".avg";
                    case NVPW_ROLLUP_OP_MAX:
                        return ".max";
                    case NVPW_ROLLUP_OP_MIN:
                        return ".min";
                    case NVPW_ROLLUP_OP_SUM:
                        return ".sum";
                    default:
                        return "";
                }
                return "";
            }

            std::string GetSubMetricString(NVPW_Submetric subMetric)
            {
                // TODO Check for peak brust missing submetrics
                switch (subMetric)
                {
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED:
                        return ".peak_sustained";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE:
                        return ".peak_sustained_active";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND:
                        return ".peak_sustained_active.per_second";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED:
                        return ".peak_sustained_elapsed";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND:
                        return ".peak_sustained_elapsed.per_second";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME:
                        return ".peak_sustained_frame";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND:
                        return ".peak_sustained_frame.per_second";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION:
                        return ".peak_sustained_region";
                    case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND:
                        return ".peak_sustained_region.per_second";
                    case NVPW_SUBMETRIC_PER_CYCLE_ACTIVE:
                        return ".per_cycle_active";
                    case NVPW_SUBMETRIC_PER_CYCLE_ELAPSED:
                        return ".per_cycle_elapsed";
                    case NVPW_SUBMETRIC_PER_CYCLE_IN_FRAME:
                        return ".per_cycle_in_frame";
                    case NVPW_SUBMETRIC_PER_CYCLE_IN_REGION:
                        return ".per_cycle_in_region";
                    case NVPW_SUBMETRIC_PER_SECOND:
                        return ".per_second";
                    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE:
                        return ".pct_of_peak_sustained_active";
                    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED:
                        return ".pct_of_peak_sustained_elapsed";
                    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME:
                        return ".pct_of_peak_sustained_frame";
                    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION:
                        return ".pct_of_peak_sustained_region";
                    case NVPW_SUBMETRIC_MAX_RATE:
                        return ".max_rate";
                    case NVPW_SUBMETRIC_PCT:
                        return ".pct";
                    case NVPW_SUBMETRIC_RATIO:
                        return ".ratio";
                    case NVPW_SUBMETRIC_NONE:
                    default:
                        return "";
                }
                return "";
            }

            bool ExportSupportedMetrics(const char* chipName,
                                        bool listSubMetrics,
                                        const uint8_t* pCounterAvailabilityImage,
                                        std::vector<std::string>& pMetricsList)
            {
                NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
                calculateScratchBufferSizeParam.pChipName = chipName;
                calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

                std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
                NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
                metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
                metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
                metricEvaluatorInitializeParams.pChipName = chipName;
                metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
                NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

                for (auto i = 0; i < NVPW_MetricType::NVPW_METRIC_TYPE__COUNT; ++i)
                {
                    NVPW_MetricType metricType = static_cast<NVPW_MetricType>(i);
                    NVPW_MetricsEvaluator_GetMetricNames_Params getMetricNamesParams = { NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE };
                    getMetricNamesParams.metricType = metricType;
                    getMetricNamesParams.pMetricsEvaluator = metricEvaluator;
                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricNames(&getMetricNamesParams));
                    
                    for (size_t metricIndex = 0; metricIndex < getMetricNamesParams.numMetrics; ++metricIndex)
                    {
                        size_t metricNameBeginIndex = getMetricNamesParams.pMetricNameBeginIndices[metricIndex];
                        for (size_t rollupOpIndex = 0; rollupOpIndex < NVPW_RollupOp::NVPW_ROLLUP_OP__COUNT; ++rollupOpIndex)
                        {
                            std::string metricName = &getMetricNamesParams.pMetricNames[metricNameBeginIndex];
                            if (metricType != NVPW_MetricType::NVPW_METRIC_TYPE_RATIO)
                            {
                                metricName += GetMetricRollupOpString((NVPW_RollupOp)rollupOpIndex);
                            }
                        
                            if (listSubMetrics)
                            {
                                NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params getSupportedSubmetricsParmas = { NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE };
                                getSupportedSubmetricsParmas.metricType = metricType;
                                getSupportedSubmetricsParmas.pMetricsEvaluator = metricEvaluator;
                                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetSupportedSubmetrics(&getSupportedSubmetricsParmas));
                                for (size_t submetricIndex = 0; submetricIndex < getSupportedSubmetricsParmas.numSupportedSubmetrics; ++submetricIndex)
                                {
                                    std::string submetricName = std::string(metricName) + GetSubMetricString((NVPW_Submetric)getSupportedSubmetricsParmas.pSupportedSubmetrics[submetricIndex]);
                                    pMetricsList.push_back(submetricName);
                                }
                            }
                        }
                    }
                }

                NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
                metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
                return true;
            }

            bool ListSupportedChips()
            {
                NVPW_GetSupportedChipNames_Params getSupportedChipNames = { NVPW_GetSupportedChipNames_Params_STRUCT_SIZE };
                RETURN_IF_NVPW_ERROR(false, NVPW_GetSupportedChipNames(&getSupportedChipNames));
                std::cout << "\nNumber of supported chips : " << getSupportedChipNames.numChipNames;
                std::cout << "\nList of supported chips : \n";

                for (size_t i = 0; i < getSupportedChipNames.numChipNames; i++)
                {
                    std::cout << " " << getSupportedChipNames.ppChipNames[i] << "\t";
                }
                std::cout << "\n";
                return true;
            }

            bool ListMetrics(const char* chip, bool listSubMetrics, const uint8_t* pCounterAvailabilityImage) 
            {
                std::vector<std::string> metricsList;
                if (ExportSupportedMetrics(chip, listSubMetrics, pCounterAvailabilityImage, metricsList))
                {
                    std::cout << metricsList.size() << " metrics in total on the chip\nMetrics List : \n";
                    for (auto metric : metricsList)
                    {
                        std::cout << metric << "\n";
                    }
                    return true;
                }
                return false;
            }
        
        }
    }
}
