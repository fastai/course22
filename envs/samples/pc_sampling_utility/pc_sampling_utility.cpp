#include "pc_sampling_utility_helper.h"
#include <stdlib.h>

int
main(
    int argc,
    char *argv[])
{
    CUpti_PCSamplingData *pMergedPcSampDataBuffer = NULL;
    size_t numMergedPcSampDataBuffer = 0;

    Init();
    ParseCommandLineArgs(argc, argv);
    FillCrcModuleMap();
    RetrievePcSampData();

    if (!disableSourceCorrelation)
    {
        if (!disableMerge && collectionMode != CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED)
        {
            MergePcSampDataBuffers(&pMergedPcSampDataBuffer, numMergedPcSampDataBuffer);
            SourceCorrelation(pMergedPcSampDataBuffer, numMergedPcSampDataBuffer);
        }
        else
        {
            SourceCorrelation(buffersRetrievedDataVector.data(), buffersRetrievedDataVector.size());
        }
    }
    else
    {
        if (!disablePcInfoPrints)
        {
            PrintRetrievedPcSampData();
        }
    }

    // Free memory
    FreePcSampStallReasonsMemory();
    FreePcSampDataBuffers(pMergedPcSampDataBuffer, numMergedPcSampDataBuffer);
    FreePcSampDataBuffers(buffersRetrievedDataVector.data(), buffersRetrievedDataVector.size());
    FreeCrcModuleMapMemory();

    exit(EXIT_SUCCESS);
}
