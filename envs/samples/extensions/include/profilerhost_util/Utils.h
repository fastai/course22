#include <nvperf_host.h>

#define RETURN_IF_NVPW_ERROR(retval, actual)                                        \
do {                                                                                \
    NVPA_Status status = actual;                                                    \
    if (NVPA_STATUS_SUCCESS != status) {                                            \
        fprintf(stderr, "FAILED: %s with error %s\n", #actual, NV::Metric::Utils::GetNVPWResultString(status)); \
        return retval;                                                              \
    }                                                                               \
} while (0)

namespace NV {
    namespace Metric {
        namespace Utils {

            static const char* GetNVPWResultString(NVPA_Status status) {
                const char* errorMsg = NULL;
                switch (status)
                {
                case NVPA_STATUS_ERROR:
                    errorMsg = "NVPA_STATUS_ERROR";
                    break;
                case NVPA_STATUS_INTERNAL_ERROR:
                    errorMsg = "NVPA_STATUS_INTERNAL_ERROR";
                    break;
                case NVPA_STATUS_NOT_INITIALIZED:
                    errorMsg = "NVPA_STATUS_NOT_INITIALIZED";
                    break;
                case NVPA_STATUS_NOT_LOADED:
                    errorMsg = "NVPA_STATUS_NOT_LOADED";
                    break;
                case NVPA_STATUS_FUNCTION_NOT_FOUND:
                    errorMsg = "NVPA_STATUS_FUNCTION_NOT_FOUND";
                    break;
                case NVPA_STATUS_NOT_SUPPORTED:
                    errorMsg = "NVPA_STATUS_NOT_SUPPORTED";
                    break;
                case NVPA_STATUS_NOT_IMPLEMENTED:
                    errorMsg = "NVPA_STATUS_NOT_IMPLEMENTED";
                    break;
                case NVPA_STATUS_INVALID_ARGUMENT:
                    errorMsg = "NVPA_STATUS_INVALID_ARGUMENT";
                    break;
                case NVPA_STATUS_INVALID_METRIC_ID:
                    errorMsg = "NVPA_STATUS_INVALID_METRIC_ID";
                    break;
                case NVPA_STATUS_DRIVER_NOT_LOADED:
                    errorMsg = "NVPA_STATUS_DRIVER_NOT_LOADED";
                    break;
                case NVPA_STATUS_OUT_OF_MEMORY:
                    errorMsg = "NVPA_STATUS_OUT_OF_MEMORY";
                    break;
                case NVPA_STATUS_INVALID_THREAD_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_THREAD_STATE";
                    break;
                case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
                    errorMsg = "NVPA_STATUS_FAILED_CONTEXT_ALLOC";
                    break;
                case NVPA_STATUS_UNSUPPORTED_GPU:
                    errorMsg = "NVPA_STATUS_UNSUPPORTED_GPU";
                    break;
                case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION";
                    break;
                case NVPA_STATUS_OBJECT_NOT_REGISTERED:
                    errorMsg = "NVPA_STATUS_OBJECT_NOT_REGISTERED";
                    break;
                case NVPA_STATUS_INSUFFICIENT_PRIVILEGE:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_PRIVILEGE";
                    break;
                case NVPA_STATUS_INVALID_CONTEXT_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_CONTEXT_STATE";
                    break;
                case NVPA_STATUS_INVALID_OBJECT_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_OBJECT_STATE";
                    break;
                case NVPA_STATUS_RESOURCE_UNAVAILABLE:
                    errorMsg = "NVPA_STATUS_RESOURCE_UNAVAILABLE";
                    break;
                case NVPA_STATUS_DRIVER_LOADED_TOO_LATE:
                    errorMsg = "NVPA_STATUS_DRIVER_LOADED_TOO_LATE";
                    break;
                case NVPA_STATUS_INSUFFICIENT_SPACE:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_SPACE";
                    break;
                case NVPA_STATUS_OBJECT_MISMATCH:
                    errorMsg = "NVPA_STATUS_OBJECT_MISMATCH";
                    break;
                case NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED:
                    errorMsg = "NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED";
                    break;
                default:
                    break;
                }

                return errorMsg;
            }
        }
    }
}