Copyright 2021 NVIDIA Corporation. All rights reserved

Profiling API injection sample code

Build this sample with

make CUDA_INSTALL_PATH=/path/to/cuda

This x86 linux-only sample contains 3 build targets:

libinjection.so
    * When CUDA_INJECTION64_PATH is set to a shared library, at initialization, CUDA
      will load the shared object and call the function named 'InitializeInjection'.
      At this point it is valid to call CUPTI routines to register Callbacks, enable
      Activity tracing, etc.
    * Registers callbacks for cuLaunchKernel and context creation.  This will be
      sufficient for many target applications, but others may require other launches
      to be matched, eg cuLaunchCoooperativeKernel or cuLaunchGrid.  See the Callback
      API for all possible kernel launch callbacks.
    * Creates a Profiler API configuration for each context in the target (using the
      context creation callback).  The Profiler API is configured using Kernel Replay
      and Auto Range modes with a configurable number of kernel launches within a pass.
    * The kernel launch callback is used to track how many kernels have launched in
      a given context's current pass, and if the pass reached its maximum count, it
      prints the metrics and starts a new pass.
    * At exit, any context with an unprocessed metrics (any which had partially
      completed a pass) print their data.
    * This library links in the profilerHostUtils library which may be built from the
      cuda/extras/CUPTI/samples/extensions/src/profilerhost_util/ directory
    * The sample uses the following environment variables:
        ** INJECTION_KERNEL_COUNT: Since injection doesn't know how many kernels a target
                application may run, it must pick a number of kernels to run in a single
                session, and once this many kernels run, it ends the session and restarts.
                This sets the number of kernels in a session, defaulting to 10.
        ** INJECTION_METRICS: This sets the metrics to gather, separated by space, comma,
                or semicolon.  Default metrics are:
                    sm__cycles_elapsed.avg
                    smsp__sass_thread_inst_executed_op_dadd_pred_on.avg
                    smsp__sass_thread_inst_executed_op_dfma_pred_on.avg

simple_target
    * Very simple executable which calls a kernel several times with increasing amount
      of work per call.

complex_target
    * More complicated example (similar to the concurrent_profiling sample) which
      launches several patterns of kernels - using default stream, multiple streams,
      and multiple devices if there are more than one device.

To use the injection library, set CUDA_INJECTION64_PATH to point to that library
when you launch the target application:

env CUDA_INJECTION64_PATH=./libinjection.so ./simple_target
