#pragma once

#include <string>
#include <vector>
namespace NV {
    namespace Metric {
        namespace Eval {
            struct MetricNameValue
            {
                std::string metricName;
                int numRanges;
                // <rangeName , metricValue> pair
                std::vector < std::pair<std::string, double> > rangeNameMetricValueMap;
            };

            /* Function to get aggregate metric values
             * @param[in]  chipName                 Chip name for which to get metric values
             * @param[in]  counterDataImage         Counter data image
             * @param[in]  metricNames              List of metrics to read from counter data image
             * @param[out] metricNameValueMap       Metric name value map
             * @param[in] pCounterAvailabilityImage  Pointer to counter availability image queried on target device
             */
            bool GetMetricGpuValue( std::string chipName,
                                    const std::vector<uint8_t>& counterDataImage,
                                    const std::vector<std::string>& metricNames,
                                    std::vector<MetricNameValue>& metricNameValueMap,
                                    const uint8_t* pCounterAvailabilityImage = NULL);

            /* Function to print aggregate metric values
             * @param[in]   chipName                    Chip name for which to get metric values
             * @param[in]   counterDataImage            Counter data image
             * @param[in]   metricNames                 List of metrics to read from counter data image
             * @param[in]   pCounterAvailabilityImage   Pointer to counter availability image queried on target device
             */
            bool PrintMetricValues( std::string chipName,
                                    const std::vector<uint8_t>& counterDataImage,
                                    const std::vector<std::string>& metricNames,
                                    const uint8_t* pCounterAvailabilityImage = NULL);
            }
    }
}