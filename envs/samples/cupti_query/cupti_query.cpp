/*
 * Copyright 2010-2019 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to query domains and events supported by device
 *
 */

// System headers
#include <stdio.h>
#include <stdlib.h>

// CUDA headers
#include <cuda.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti.h>

// Global variables
static unsigned int s_OptionsFlag = 0;

// Macros
#define NAME_SHORT  64
#define NAME_LONG   128
#define DESC_SHORT  512
#define DESC_LONG   2048

#define CATEGORY_LENGTH         sizeof(CUpti_EventCategory)
#define SetOptionsFlag(bit)     (s_OptionsFlag |= (1<<bit))
#define IsOptionsFlagSet(bit)   ((s_OptionsFlag & (1<<bit)) ? 1 : 0)


enum EnumOptions
{
    FLAG_DEVICE_ID = 0,
    FLAG_DOMAIN_ID,
    FLAG_GET_DOMAINS,
    FLAG_GET_EVENTS,
    FLAG_GET_METRICS
};

typedef struct DomainData_st
{
    // Domain id.
    CUpti_EventDomainID domainId;
    // Domain name.
    char domainName[NAME_SHORT];
    // Number of domain instances. (profiled)
    uint32_t profiledInstanceCnt;
    // Number of domain instances. (total)
    uint32_t totalInstanceCnt;
    CUpti_EventCollectionMethod eventCollectionMethod;
} DomainData;

typedef union
{
    // Event id.
    CUpti_EventID eventId;
    // Metric id.
    CUpti_MetricID metricId;
} CuptiId;

typedef struct EventData_st
{
    CuptiId Id;
    // Event name.
    char eventName[NAME_SHORT];
    // Short description of the event.
    char shortDesc[DESC_SHORT];
    // Long description of the event.
    char longDesc[DESC_LONG];
    // Category of the event.
    CUpti_EventCategory  category;
} EventData;

// Functions
static void
PrintUsage()
{
    printf("Usage: cupti_query\n");
    printf("       -help                                            : Displays help message.\n");
    printf("       -device <dev_id> -getdomains                     : Displays supported domains for specified device.\n");
    printf("       -device <dev_id> -getmetrics                     : Displays supported metrics for specified device.\n");
    printf("       -device <dev_id> -domain <domain_id> -getevents  : Displays supported events for specified domain and device.\n");
    printf("Note: Default device is 0 and default domain is first domain for device.\n");
}

// Add a null terminator to the end of a string if the string.
// Length equals the maximum length.
// (As in that case there was no room to write the null terminator.)
static void
CheckNullTerminator(
    char *pString,
    size_t length,
    size_t maxLength)
{
    if (length >= maxLength)
    {
        pString[maxLength - 1] = '\0';
    }
}

int
EnumEventDomains(
    CUdevice dev)
{
    CUptiResult cuptiResult = CUPTI_SUCCESS;
    CUpti_EventDomainID *pDomainId = NULL;
    DomainData domainData;
    uint32_t maxDomains = 0, i = 0;
    size_t size = 0;

    CUPTI_API_CALL(cuptiDeviceGetNumEventDomains(dev, &maxDomains));

    if (maxDomains == 0)
    {
        printf("Error: No domain is exposed by dev = %d\n", dev);
        cuptiResult = CUPTI_ERROR_UNKNOWN;
        goto Exit;
    }

    size = sizeof(CUpti_EventDomainID) * maxDomains;
    pDomainId = (CUpti_EventDomainID *)malloc(size);
    MEMORY_ALLOCATION_CALL(pDomainId);
    memset(pDomainId, 0, size);

    CUPTI_API_CALL(cuptiDeviceEnumEventDomains(dev, &size, pDomainId));

    // Enum domains.
    for (i = 0; i < maxDomains; i++)
    {
        domainData.domainId = pDomainId[i];

        // Query domain name.
        size = NAME_SHORT;
        CUPTI_API_CALL(cuptiEventDomainGetAttribute(domainData.domainId, CUPTI_EVENT_DOMAIN_ATTR_NAME, &size, (void*)domainData.domainName));
        CheckNullTerminator(domainData.domainName, size, NAME_SHORT);

        // Query number of profiled instances in the domain.
        size = sizeof(domainData.profiledInstanceCnt);
        CUPTI_API_CALL(cuptiDeviceGetEventDomainAttribute(dev, domainData.domainId, CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT, &size, (void *)&domainData.profiledInstanceCnt));

        // Query total instances in the domain.
        size = sizeof(domainData.totalInstanceCnt);
        CUPTI_API_CALL(cuptiDeviceGetEventDomainAttribute(dev, domainData.domainId, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &size, (void *)&domainData.totalInstanceCnt));

        size = sizeof(CUpti_EventCollectionMethod);
        CUPTI_API_CALL(cuptiEventDomainGetAttribute(domainData.domainId, CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD, &size, (void *)&domainData.eventCollectionMethod));

        printf ("Domain# %u\n", i+1);
        printf ("Id         = %d\n",   domainData.domainId);
        printf ("Name       = %s\n",   domainData.domainName);
        printf ("Profiled instance count = %u\n", domainData.profiledInstanceCnt);
        printf ("Total instance count = %u\n", domainData.totalInstanceCnt);

        printf ("Event collection method = ");
        switch (domainData.eventCollectionMethod)
        {
            case CUPTI_EVENT_COLLECTION_METHOD_PM:
                printf("CUPTI_EVENT_COLLECTION_METHOD_PM\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_SM:
                printf("CUPTI_EVENT_COLLECTION_METHOD_SM\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED:
                printf("CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED\n");
                break;
            case CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC:
                printf("CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC\n");
                break;
            default:
                printf("\nError: Invalid event collection method!\n");
                return -1;
        }
    }

Exit:
    if (pDomainId)
    {
        free(pDomainId);
    }

    if (cuptiResult == CUPTI_SUCCESS)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int
EnumEvents(
    CUpti_EventDomainID domainId)
{
    EventData eventData;
    CUptiResult cuptiResult = CUPTI_SUCCESS;
    CUpti_EventID *pEventId = NULL;
    uint32_t maxEvents = 0;
    uint32_t i = 0;
    size_t size = 0;

    // Query number of events available in the domain.
    CUPTI_API_CALL(cuptiEventDomainGetNumEvents(domainId, &maxEvents));

    size = sizeof(CUpti_EventID) * maxEvents;
    pEventId = (CUpti_EventID *)malloc(size);
    MEMORY_ALLOCATION_CALL(pEventId);
    memset(pEventId, 0, size);

    CUPTI_API_CALL(cuptiEventDomainEnumEvents(domainId, &size, pEventId));

    // Query event information.
    for (i = 0; i < maxEvents; i++)
    {
        eventData.Id.eventId = pEventId[i];

        size = NAME_SHORT;
        CUPTI_API_CALL(cuptiEventGetAttribute(eventData.Id.eventId, CUPTI_EVENT_ATTR_NAME, &size, (uint8_t *)eventData.eventName));
        CheckNullTerminator(eventData.eventName, size, NAME_SHORT);

        size = DESC_SHORT;
        CUPTI_API_CALL(cuptiEventGetAttribute(eventData.Id.eventId, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size, (uint8_t *)eventData.shortDesc));
        CheckNullTerminator(eventData.shortDesc, size, DESC_SHORT);

        size = DESC_LONG;
        CUPTI_API_CALL(cuptiEventGetAttribute(eventData.Id.eventId, CUPTI_EVENT_ATTR_LONG_DESCRIPTION, &size, (uint8_t *)eventData.longDesc));
        CheckNullTerminator(eventData.longDesc, size, DESC_LONG);

        size = CATEGORY_LENGTH;
        CUPTI_API_CALL(cuptiEventGetAttribute(eventData.Id.eventId, CUPTI_EVENT_ATTR_CATEGORY, &size, &eventData.category));

        printf("Event# %u\n", i+1);
        printf("Id        = %d\n", eventData.Id.eventId);
        printf("Name      = %s\n", eventData.eventName);
        printf("Shortdesc = %s\n", eventData.shortDesc);
        printf("Longdesc  = %s\n", eventData.longDesc);

        switch (eventData.category)
        {
            case CUPTI_EVENT_CATEGORY_INSTRUCTION:
                printf("Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_MEMORY:
                printf("Category  = CUPTI_EVENT_CATEGORY_MEMORY\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_CACHE:
                printf("Category  = CUPTI_EVENT_CATEGORY_CACHE\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER:
                printf("Category  = CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER\n\n");
                break;
            case CUPTI_EVENT_CATEGORY_SYSTEM:
                printf("Category  = CUPTI_EVENT_CATEGORY_SYSTEM\n\n");
                break;
            default:
                printf("\nInvalid category!\n");
        }

    }

    if (pEventId)
    {
        free(pEventId);
    }

    if (cuptiResult == CUPTI_SUCCESS)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int EnumMetrics(
    CUdevice dev)
{
    EventData metricData;
    CUptiResult cuptiResult = CUPTI_SUCCESS;
    CUpti_MetricID *pMetricId = NULL;
    uint32_t maxMetrics = 0;
    uint32_t i = 0;
    size_t size = 0;

    CUPTI_API_CALL(cuptiDeviceGetNumMetrics(dev, &maxMetrics));

    size = sizeof(CUpti_EventID) * maxMetrics;
    pMetricId = (CUpti_MetricID *)malloc(size);
    MEMORY_ALLOCATION_CALL(pMetricId);
    memset(pMetricId, 0, size);

    CUPTI_API_CALL(cuptiDeviceEnumMetrics(dev, &size, pMetricId));

    // query metric info
    for (i = 0; i < maxMetrics; i++)
    {
        metricData.Id.metricId = pMetricId[i];

        size = NAME_SHORT;
        CUPTI_API_CALL(cuptiMetricGetAttribute(metricData.Id.metricId, CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *)metricData.eventName));
        CheckNullTerminator(metricData.eventName, size, NAME_SHORT);

        size = DESC_SHORT;
        CUPTI_API_CALL(cuptiMetricGetAttribute(metricData.Id.metricId, CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &size, (uint8_t *)metricData.shortDesc));
        CheckNullTerminator(metricData.shortDesc, size, DESC_SHORT);

        size = DESC_LONG;
        CUPTI_API_CALL(cuptiMetricGetAttribute(metricData.Id.metricId, CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size, (uint8_t *)metricData.longDesc));
        CheckNullTerminator(metricData.longDesc, size, DESC_LONG);

        printf("Metric# %u\n", i + 1);
        printf("Id        = %d\n", metricData.Id.metricId);
        printf("Name      = %s\n", metricData.eventName);
        printf("Shortdesc = %s\n", metricData.shortDesc);
        printf("Longdesc  = %s\n\n", metricData.longDesc);
    }

    if (pMetricId)
    {
        free(pMetricId);
    }

    if (cuptiResult == CUPTI_SUCCESS)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

void ParseCommandLineArgs(int argc, char *argv[], int &deviceId, CUpti_EventDomainID &domainId)
{
    for (int k = 1; k < argc; k++)
    {
        if ((k + 1 < argc) &&
            stricmp(argv[k], "-device") == 0)
        {
            deviceId = atoi(argv[k+1]);
            SetOptionsFlag(FLAG_DEVICE_ID);
            k++;
        }
        else if ((k+1 < argc) &&
                stricmp(argv[k], "-domain") == 0)
        {
            domainId = (CUpti_EventDomainID)atoi(argv[k + 1]);
            SetOptionsFlag(FLAG_DOMAIN_ID);
            k++;
        }
        else if ((k < argc) &&
                stricmp(argv[k], "-getdomains") == 0)
        {
            SetOptionsFlag(FLAG_GET_DOMAINS);
        }
        else if (stricmp(argv[k], "-getevents") == 0)
        {
            SetOptionsFlag(FLAG_GET_EVENTS);
        }
        else if (stricmp(argv[k], "-getmetrics") == 0)
        {
            SetOptionsFlag(FLAG_GET_METRICS);
        }
        else if ((stricmp(argv[k], "--help") == 0) ||
                 (stricmp(argv[k], "-help") == 0) ||
                 (stricmp(argv[k], "-h") == 0))
        {
            PrintUsage();
            exit(EXIT_SUCCESS);
        }
        else
        {
            printf("Warning: Invalid/Incomplete option %s.\n", argv[k]);
        }
    }
}

int
main(
    int argc,
    char *argv[])
{
    CUdevice dev;
    CUptiResult cuptiResult = CUPTI_SUCCESS;
    int ret = 0;
    int deviceId = 0;
    int deviceCount = 0;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    char deviceName[256];
    CUpti_EventDomainID domainId = 0;
    size_t size = 0;

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\n");
        ret = -2;
        goto Exit;
    }

    // Parse command line arguments.
    ParseCommandLineArgs(argc, argv, deviceId, domainId);

    if (!IsOptionsFlagSet(FLAG_DEVICE_ID)) {
        // Default is device 0.
        printf("Assuming default device id 0\n");
        deviceId = 0;
    }

    // Show events if no explicit flag is set.
    if (!IsOptionsFlagSet(FLAG_GET_DOMAINS) &&
        !IsOptionsFlagSet(FLAG_GET_EVENTS) &&
        !IsOptionsFlagSet(FLAG_GET_METRICS))
    {
        SetOptionsFlag(FLAG_GET_EVENTS);
    }

    DRIVER_API_CALL(cuDeviceGet(&dev, deviceId));

    DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, dev));

    printf("CUDA Device Id  : %d\n", deviceId);
    printf("CUDA Device Name: %s\n\n", deviceName);

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    if (IsOptionsFlagSet(FLAG_GET_DOMAINS))
    {
        if (EnumEventDomains(dev))
        {
            printf("Error: EnumEventDomains failed.\n");
            ret = -1;
            goto Exit;
        }
    }
    else if (IsOptionsFlagSet(FLAG_GET_EVENTS))
    {
        if (!IsOptionsFlagSet(FLAG_DOMAIN_ID))
        {
            // Query first domain on the device.
            size = sizeof(CUpti_EventDomainID);
            CUPTI_API_CALL(cuptiDeviceEnumEventDomains(dev, &size, (CUpti_EventDomainID *)&domainId));

            printf("Assuming default domain id %d.\n", domainId);
        }
        else
        {
            // Validate the domain on the device.
            CUpti_EventDomainID *pDomainIdArray = NULL;
            uint32_t maxDomains = 0, i = 0;

            CUPTI_API_CALL(cuptiDeviceGetNumEventDomains(dev, &maxDomains));

            if (maxDomains == 0)
            {
                printf("Warning: No domain is exposed by dev = %d\n", dev);
                cuptiResult = CUPTI_ERROR_UNKNOWN;
                ret = -2;
                goto Exit;
            }

            size = sizeof(CUpti_EventDomainID) * maxDomains;
            pDomainIdArray = (CUpti_EventDomainID *)malloc(size);
            MEMORY_ALLOCATION_CALL(pDomainIdArray);
            memset(pDomainIdArray, 0, size);

            // Enum domains.
            CUPTI_API_CALL(cuptiDeviceEnumEventDomains(dev, &size, pDomainIdArray));

            for (i = 0; i < maxDomains; i++)
            {
                if (pDomainIdArray[i] == domainId)
                {
                    break;
                }
            }
            free(pDomainIdArray);

            if (i == maxDomains)
            {
                printf("Warning: Domain Id %d is not supported by device.\n", domainId);
                ret = -2;
                goto Exit;
            }
        }

        if (EnumEvents(domainId))
        {
            printf("Error: EnumEvents failed.\n");
            ret = -1;
            goto Exit;
        }
    }
    else if (IsOptionsFlagSet(FLAG_GET_METRICS))
    {
        if(EnumMetrics(dev)) {
            printf("Error: EnumMetrics() failed.\n");
            ret = -1;
            goto Exit;
        }
    }

Exit:
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    if(ret == -1)
    {
        exit(EXIT_FAILURE);
    }
    else if (ret == -2)
    {
        exit(EXIT_WAIVED);
    }
    else
    {
        exit(EXIT_SUCCESS);
    }
}
