
1. Build the sample
    1.1 On your machine, go to the Cupti_trace_injection sample directory.
    1.2 A template Makefile is present which will let you build the sample out of the box.
    1.3 For windows need to build detours library from source. Refer following steps to build it.
    	a. Download source code from https://github.com/microsoft/Detours
    	b. Unzip and go to Detours folder
    	c. set DETOURS_TARGET_PROCESSOR=X64
    	d. "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
    	e. NMAKE
    	f. copy detours.h and detours.lib in sample folder.
    1.4 Run make. For linux the dynamic library libcupti_trace_injection.so and for windows libcupti_trace_injection.dll should be created in the folder.

2. Run the CUDA application after setting the environment variable CUDA_INJECTION64_PATH to the path of the injection library.
   When CUDA_INJECTION64_PATH is set to a shared library, at initialization, CUDA will load the shared object and call the function named 'InitializeInjection'.
   CUDA application needs not to be modified.

    2.1  On Linux,
            $ export CUDA_INJECTION64_PATH=<full_path>/libcupti_trace_injection.so
            $ <run CUDA application>
    2.2  On Windows,
            > set CUDA_INJECTION64_PATH=<full_path>/libcupti_trace_injection.dll
            > <run CUDA application>


