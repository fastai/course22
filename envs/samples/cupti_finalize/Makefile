#
# Copyright 2020-2021 NVIDIA Corporation. All rights reserved
# This sample is for Linux desktop platforms. (Power, x86_64, ARM Server)
#

ifndef OS
OS   := $(shell uname)
HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    INCLUDES = -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
    INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
    TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
else
    INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I../../include
    EXTRAS_LIB_PATH := ../../lib64
    LIB_PATH ?= "$(CUDA_INSTALL_PATH)/lib64"
endif

INCLUDES += -I../common

LIBS :=
ifeq ($(HOST_ARCH), $(TARGET_ARCH))
    LIBS = -L $(EXTRAS_LIB_PATH)
endif
LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti

NVCCFLAGS :=
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

FINALIZE_LIB = libCuptiFinalize.so

cupti_finalize: cupti_finalize.cpp
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(INCLUDES) -o $(FINALIZE_LIB) -Xcompiler -fPIC -shared $< $(LIBS)
	$(info export CUDA_INJECTION64_PATH=<full_path>/libCuptiFinalize.so and run any CUDA sample with runtime more than 10 seconds(Graphics/Simulations samples) for demonstration)
clean:
	rm -f $(FINALIZE_LIB) cupti_finalize.o
