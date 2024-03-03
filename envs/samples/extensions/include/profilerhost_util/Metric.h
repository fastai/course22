#pragma once

#include <string>
#include <vector>
namespace NV {
    namespace Metric {
        namespace Config {
            /* Function to get Config image
            * @param[in]  chipName                          Chip name for which configImage is to be generated
            * @param[in]  metricNames                       List of metrics for which configImage is to be generated
            * @param[out] configImage                       Generated configImage
            * @param[in]  pCounterAvailabilityImage         Pointer to counter availability image queried on target device, can be used to filter unavailable metrics
            */
            bool GetConfigImage(std::string chipName,
                                const std::vector<std::string>& metricNames,
                                std::vector<uint8_t>& configImage,
                                const uint8_t* pCounterAvailabilityImage = NULL);

            /* Function to get CounterDataPrefix image
            * @param[in]  chipName                  Chip name for which counterDataImagePrefix is to be generated
            * @param[in]  metricNames               List of metrics for which counterDataImagePrefix is to be generated
            * @param[out] counterDataImagePrefix    Generated counterDataImagePrefix
            * @param[in] pCounterAvailabilityImage  Pointer to counter availability image queried on target device, can be used to filter unavailable metrics
            */
            bool GetCounterDataPrefixImage(std::string chipName,
                                           const std::vector<std::string>& metricNames,
                                           std::vector<uint8_t>& counterDataImagePrefix,
                                           const uint8_t* pCounterAvailabilityImage = NULL);
        }
    }
}
