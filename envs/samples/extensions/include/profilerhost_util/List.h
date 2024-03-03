#pragma once

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>

namespace NV {
    namespace Metric {
        namespace Enum {
            
            bool ListSupportedChips();

            /* Function to print list of all metrics for a given chip
             * @param[in]  chipName                     Chip Name for which metrics are to be listed
             * @param[in]  listSubMetrics               Whether submetrics(Peak, PerCycle, PctOfPeak) are to be listed or not
             * @param[in]  pCounterAvailabilityImage    Pointer to counter availability image queried on target device 
             */
            bool ListMetrics(const char* chipName,
                             bool listSubMetrics,
                             const uint8_t* pCounterAvailabilityImage = NULL);

            /* Function to export list of all metrics for a given chip
             * @param[in]   chipName                    Chip Name for which metrics are to be listed
             * @param[in]   listSubMetrics              Whether submetrics(Peak, PerCycle, PctOfPeak) are to be listed or not
             * @param[in]   pCounterAvailabilityImage   Pointer to counter availability image queried on target device 
             * @param[out]  pMetricsList                Supported Metrics list for given chip
             */
            bool ExportSupportedMetrics(const char* chipName,
                                        bool listSubMetrics,
                                        const uint8_t* pCounterAvailabilityImage,
                                        std::vector<std::string>& pMetricsList);
        }
    }
}