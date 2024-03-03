// This file is generated.  Any changes you make will be lost during the next clean build.

// CUDA public interface, for type definitions and api function prototypes
#include "cuda_gl_interop.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Currently used parameter trace structures
typedef struct cudaGLGetDevices_v4010_params_st {
    unsigned int *pCudaDeviceCount;
    int *pCudaDevices;
    unsigned int cudaDeviceCount;
    enum cudaGLDeviceList deviceList;
} cudaGLGetDevices_v4010_params;

typedef struct cudaGraphicsGLRegisterImage_v3020_params_st {
    struct cudaGraphicsResource **resource;
    GLuint image;
    GLenum target;
    unsigned int flags;
} cudaGraphicsGLRegisterImage_v3020_params;

typedef struct cudaGraphicsGLRegisterBuffer_v3020_params_st {
    struct cudaGraphicsResource **resource;
    GLuint buffer;
    unsigned int flags;
} cudaGraphicsGLRegisterBuffer_v3020_params;

typedef struct cudaGLSetGLDevice_v3020_params_st {
    int device;
} cudaGLSetGLDevice_v3020_params;

typedef struct cudaGLRegisterBufferObject_v3020_params_st {
    GLuint bufObj;
} cudaGLRegisterBufferObject_v3020_params;

typedef struct cudaGLMapBufferObject_v3020_params_st {
    void **devPtr;
    GLuint bufObj;
} cudaGLMapBufferObject_v3020_params;

typedef struct cudaGLUnmapBufferObject_v3020_params_st {
    GLuint bufObj;
} cudaGLUnmapBufferObject_v3020_params;

typedef struct cudaGLUnregisterBufferObject_v3020_params_st {
    GLuint bufObj;
} cudaGLUnregisterBufferObject_v3020_params;

typedef struct cudaGLSetBufferObjectMapFlags_v3020_params_st {
    GLuint bufObj;
    unsigned int flags;
} cudaGLSetBufferObjectMapFlags_v3020_params;

typedef struct cudaGLMapBufferObjectAsync_v3020_params_st {
    void **devPtr;
    GLuint bufObj;
    cudaStream_t stream;
} cudaGLMapBufferObjectAsync_v3020_params;

typedef struct cudaGLUnmapBufferObjectAsync_v3020_params_st {
    GLuint bufObj;
    cudaStream_t stream;
} cudaGLUnmapBufferObjectAsync_v3020_params;

// Parameter trace structures for removed functions


// End of parameter trace structures
