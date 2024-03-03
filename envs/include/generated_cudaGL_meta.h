// This file is generated.  Any changes you make will be lost during the next clean build.

// Dependent includes
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// CUDA public interface, for type definitions and cu* function prototypes
#include "cudaGL.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct cuGraphicsGLRegisterBuffer_params_st {
    CUgraphicsResource *pCudaResource;
    GLuint buffer;
    unsigned int Flags;
} cuGraphicsGLRegisterBuffer_params;

typedef struct cuGraphicsGLRegisterImage_params_st {
    CUgraphicsResource *pCudaResource;
    GLuint image;
    GLenum target;
    unsigned int Flags;
} cuGraphicsGLRegisterImage_params;

typedef struct cuGLGetDevices_v2_params_st {
    unsigned int *pCudaDeviceCount;
    CUdevice *pCudaDevices;
    unsigned int cudaDeviceCount;
    CUGLDeviceList deviceList;
} cuGLGetDevices_v2_params;

typedef struct cuGLCtxCreate_v2_params_st {
    CUcontext *pCtx;
    unsigned int Flags;
    CUdevice device;
} cuGLCtxCreate_v2_params;

typedef struct cuGLRegisterBufferObject_params_st {
    GLuint buffer;
} cuGLRegisterBufferObject_params;

typedef struct cuGLMapBufferObject_v2_ptds_params_st {
    CUdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
} cuGLMapBufferObject_v2_ptds_params;

typedef struct cuGLUnmapBufferObject_params_st {
    GLuint buffer;
} cuGLUnmapBufferObject_params;

typedef struct cuGLUnregisterBufferObject_params_st {
    GLuint buffer;
} cuGLUnregisterBufferObject_params;

typedef struct cuGLSetBufferObjectMapFlags_params_st {
    GLuint buffer;
    unsigned int Flags;
} cuGLSetBufferObjectMapFlags_params;

typedef struct cuGLMapBufferObjectAsync_v2_ptsz_params_st {
    CUdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
    CUstream hStream;
} cuGLMapBufferObjectAsync_v2_ptsz_params;

typedef struct cuGLUnmapBufferObjectAsync_params_st {
    GLuint buffer;
    CUstream hStream;
} cuGLUnmapBufferObjectAsync_params;

typedef struct cuGLGetDevices_params_st {
    unsigned int *pCudaDeviceCount;
    CUdevice *pCudaDevices;
    unsigned int cudaDeviceCount;
    CUGLDeviceList deviceList;
} cuGLGetDevices_params;

typedef struct cuGLMapBufferObject_v2_params_st {
    CUdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
} cuGLMapBufferObject_v2_params;

typedef struct cuGLMapBufferObjectAsync_v2_params_st {
    CUdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
    CUstream hStream;
} cuGLMapBufferObjectAsync_v2_params;

typedef struct cuGLCtxCreate_params_st {
    CUcontext *pCtx;
    unsigned int Flags;
    CUdevice device;
} cuGLCtxCreate_params;

typedef struct cuGLMapBufferObject_params_st {
    CUdeviceptr_v1 *dptr;
    unsigned int *size;
    GLuint buffer;
} cuGLMapBufferObject_params;

typedef struct cuGLMapBufferObjectAsync_params_st {
    CUdeviceptr_v1 *dptr;
    unsigned int *size;
    GLuint buffer;
    CUstream hStream;
} cuGLMapBufferObjectAsync_params;
