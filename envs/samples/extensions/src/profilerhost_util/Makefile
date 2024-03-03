#
# Copyright 2011-2018 NVIDIA Corporation. All rights reserved
#
INCLUDES = -I../../../../include -I../../../../../../include -I../../include/profilerhost_util -I../../include/c_util

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= linux

# Set required library paths.
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifeq ($(OS), Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        LIB_PATH ?= ../../../../../../targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDALIB_STUB_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := ../../../../lib64
        LIB_PATH ?= ../../../../../../lib64
    endif
endif

ifeq ($(OS), Windows_NT)
    export PATH := $(PATH):../../../../lib/x64
    LIBS = -lcuda -L $(LIB_PATH) -lnvperf_host -lnvperf_target
    OBJ = obj
    LIBEXT = lib
    LIBPREFIX =
else
    NVCC_FLAGS = --std=c++11
    ifeq ($(OS), Linux)
        NVCC_FLAGS += -Xcompiler -fPIC
    endif
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib/x64
        LIBS = -Xlinker -framework -Xlinker cuda -L ../../../../lib/x64 -lnvperf_host -lnvperf_target
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib/x64
            ifneq ($(TARGET_ARCH), x86_64)
                EXTRAS_LIB_PATH := ../../../../lib64
                LIBS = -L $(EXTRAS_LIB_PATH)
            endif
        endif
        LIBS += $(TARGET_CUDALIB_STUB_PATH) -lcuda -L $(LIB_PATH) -lnvperf_host -lnvperf_target
    endif
    OBJ = o
    LIBEXT = a
    LIBPREFIX = lib
endif

# Point to the necessary cross-compiler.

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS), qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
            NVCC_FLAGS += -D_QNX_SOURCE
            NVCC_FLAGS += --qpp-config 8.3.0,gcc_ntoaarch64le -lsocket
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER = -ccbin $(HOST_COMPILER)
    endif
endif

$(LIBPREFIX)profilerHostUtil.$(LIBEXT): List.$(OBJ) Metric.$(OBJ) Eval.$(OBJ)
	nvcc $(NVCC_COMPILER) -o $@ -lib $^ $(LIBS)

List.$(OBJ): List.cpp
	nvcc $(NVCC_COMPILER) -c $(NVCC_FLAGS) $(INCLUDES) $<

Metric.$(OBJ): Metric.cpp
	nvcc $(NVCC_COMPILER) -c $(NVCC_FLAGS) $(INCLUDES) $<

Eval.$(OBJ): Eval.cpp
	nvcc $(NVCC_COMPILER) -c $(NVCC_FLAGS) $(INCLUDES) $<

clean:
ifeq ($(OS),Windows_NT)
	del $(LIBPREFIX)profilerHostUtil.$(LIBEXT) List.$(OBJ) Metric.$(OBJ) Eval.$(OBJ)
else
	rm -f $(LIBPREFIX)profilerHostUtil.$(LIBEXT) List.$(OBJ) Metric.$(OBJ) Eval.$(OBJ)
endif
