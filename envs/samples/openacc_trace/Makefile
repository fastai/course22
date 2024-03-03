#
# Copyright 2015-2021 NVIDIA Corporation. All rights reserved
#

# CUPTI OpenACC support is only available on Linux x86_64, ppc64le and arm64 SBSA.

# This sample requires PGI compiler version HPC SDK 20.5 or later.
# Point to your PGI OpenACC installation's include directory.
# PGI compiler must be in PATH.
CUDA_INSTALL_PATH ?= ../../../..

EXTRAS_LIB_PATH := ../../lib64
LIB_PATH ?= "$(CUDA_INSTALL_PATH)/lib64"

OPENACC_INCLUDE_PATH =

HOST_ARCH := $(shell uname -m)

INCLUDES += -I../../include -I../common
INCLUDES += -I"$(CUDA_INSTALL_PATH)/include"
INCLUDES += -I$(OPENACC_INCLUDE_PATH)

PGCPP       = nvc++
PGCPP_FLAGS = -acc -ta=nvidia:cuda11.0 -Mcuda=nordc -pgf90libs
ifneq ($(HOST_ARCH),ppc64le)
        PGCPP_FLAGS += -tp=p7 -mp=nonuma
else
        PGCPP_FLAGS += -DHOST_ARCH_PPC=1
endif

export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
TRACE_LIB = libopenacc_trace.so

all: openacc_app

$(TRACE_LIB): openacc_trace.cpp
	$(PGCPP) $(INCLUDES) -DCUPTI_DIRECTIVE_SUPPORT -fPIC -shared -o $@ $<

openacc_app: $(TRACE_LIB) openacc_app.cpp
	$(PGCPP) $(INCLUDES) $(PGCPP_FLAGS)         \
	-L. -L $(EXTRAS_LIB_PATH) -L $(LIB_PATH)  -lopenacc_trace -lcupti  \
	-o $@ $^

run: $(TRACE_LIB) openacc_app
	ACC_PROFLIB=$(TRACE_LIB) ./openacc_app

clean:
	rm -f $(TRACE_LIB) openacc_app

