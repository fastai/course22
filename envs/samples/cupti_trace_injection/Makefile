#
# Copyright 2021 NVIDIA Corporation. All rights reserved
#
ifndef OS
    OS   := $(shell uname)
    HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I../../include -I../common

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    EXTRAS_LIB_PATH := ../../lib64
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
endif

# Point to the necessary cross-compiler.
NVCCFLAGS :=
ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -L $(LIB_PATH) -lcuda -lcupti -ldetours
    LIBNAME := libcupti_trace_injection.dll
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(EXTRAS_LIB_PATH) -L $(LIB_PATH) -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -L $(LIB_PATH) -lcuda -L $(EXTRAS_LIB_PATH) -lcupti
    endif
    LIBNAME := libcupti_trace_injection.so
    NVCCFLAGS += -Xcompiler -fPIC
endif

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
            NVCCFLAGS += --qpp-config 8.3.0,gcc_ntoaarch64le -lsocket
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif

all: cupti_trace_injection
cupti_trace_injection: cupti_trace_injection.cpp
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(INCLUDES) -o $(LIBNAME) -shared $< $(LIBS)
clean:
	rm -f $(LIBNAME) cupti_trace_injection.o
