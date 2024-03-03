// This file is generated.  Any changes you make will be lost during the next clean build.

// CUDA public interface, for type definitions and api function prototypes
#include "cudart_removed.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Currently used parameter trace structures
typedef struct cudaStreamDestroy_v3020_params_st {
    cudaStream_t stream;
} cudaStreamDestroy_v3020_params;

typedef struct cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000_params_st {
    int *numBlocks;
    const void *func;
    size_t numDynamicSmemBytes;
} cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000_params;

typedef struct cudaConfigureCall_v3020_params_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem  __dv;
    cudaStream_t stream  __dv;
} cudaConfigureCall_v3020_params;

typedef struct cudaSetupArgument_v3020_params_st {
    const void *arg;
    size_t size;
    size_t offset;
} cudaSetupArgument_v3020_params;

typedef struct cudaLaunch_v3020_params_st {
    const void *func;
} cudaLaunch_v3020_params;

typedef struct cudaLaunch_ptsz_v7000_params_st {
    const void *func;
} cudaLaunch_ptsz_v7000_params;

typedef struct cudaStreamSetFlags_v10200_params_st {
    cudaStream_t hStream;
    unsigned int flags;
} cudaStreamSetFlags_v10200_params;

typedef struct cudaStreamSetFlags_ptsz_v10200_params_st {
    cudaStream_t hStream;
    unsigned int flags;
} cudaStreamSetFlags_ptsz_v10200_params;

typedef struct cudaProfilerInitialize_v4000_params_st {
    const char *configFile;
    const char *outputFile;
    cudaOutputMode_t outputMode;
} cudaProfilerInitialize_v4000_params;

typedef struct cudaThreadSetLimit_v3020_params_st {
    enum cudaLimit limit;
    size_t value;
} cudaThreadSetLimit_v3020_params;

typedef struct cudaThreadGetLimit_v3020_params_st {
    size_t *pValue;
    enum cudaLimit limit;
} cudaThreadGetLimit_v3020_params;

typedef struct cudaThreadGetCacheConfig_v3020_params_st {
    enum cudaFuncCache *pCacheConfig;
} cudaThreadGetCacheConfig_v3020_params;

typedef struct cudaThreadSetCacheConfig_v3020_params_st {
    enum cudaFuncCache cacheConfig;
} cudaThreadSetCacheConfig_v3020_params;

typedef struct cudaSetDoubleForDevice_v3020_params_st {
    double *d;
} cudaSetDoubleForDevice_v3020_params;

typedef struct cudaSetDoubleForHost_v3020_params_st {
    double *d;
} cudaSetDoubleForHost_v3020_params;

typedef struct cudaCreateTextureObject_v2_v11080_params_st {
    cudaTextureObject_t *pTexObject;
    const struct cudaResourceDesc *pResDesc;
    const struct cudaTextureDesc *pTexDesc;
    const struct cudaResourceViewDesc *pResViewDesc;
} cudaCreateTextureObject_v2_v11080_params;

typedef struct cudaGetTextureObjectTextureDesc_v2_v11080_params_st {
    struct cudaTextureDesc *pTexDesc;
    cudaTextureObject_t texObject;
} cudaGetTextureObjectTextureDesc_v2_v11080_params;

typedef struct cudaBindTexture_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t size  __dv;
} cudaBindTexture_v3020_params;

typedef struct cudaBindTexture2D_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    size_t pitch;
} cudaBindTexture2D_v3020_params;

typedef struct cudaBindTextureToArray_v3020_params_st {
    const struct textureReference *texref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToArray_v3020_params;

typedef struct cudaBindTextureToMipmappedArray_v5000_params_st {
    const struct textureReference *texref;
    cudaMipmappedArray_const_t mipmappedArray;
    const struct cudaChannelFormatDesc *desc;
} cudaBindTextureToMipmappedArray_v5000_params;

typedef struct cudaUnbindTexture_v3020_params_st {
    const struct textureReference *texref;
} cudaUnbindTexture_v3020_params;

typedef struct cudaGetTextureAlignmentOffset_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
} cudaGetTextureAlignmentOffset_v3020_params;

typedef struct cudaGetTextureReference_v3020_params_st {
    const struct textureReference **texref;
    const void *symbol;
} cudaGetTextureReference_v3020_params;

typedef struct cudaBindSurfaceToArray_v3020_params_st {
    const struct surfaceReference *surfref;
    cudaArray_const_t array;
    const struct cudaChannelFormatDesc *desc;
} cudaBindSurfaceToArray_v3020_params;

typedef struct cudaGetSurfaceReference_v3020_params_st {
    const struct surfaceReference **surfref;
    const void *symbol;
} cudaGetSurfaceReference_v3020_params;

typedef struct cudaGraphInstantiate_v10000_params_st {
    cudaGraphExec_t *pGraphExec;
    cudaGraph_t graph;
    cudaGraphNode_t *pErrorNode;
    char *pLogBuffer;
    size_t bufferSize;
} cudaGraphInstantiate_v10000_params;

// Parameter trace structures for removed functions


// End of parameter trace structures
