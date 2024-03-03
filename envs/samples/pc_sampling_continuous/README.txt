
1. Build the sample
    1.1 On your machine, go to the pc_sampling_continuous sample directory.
    1.2 A template Makefile is present which will let you build the sample out of the box.
    1.3 For windows need to build detours library from source. Refer following steps to build it.
    	a. Download source code from https://github.com/microsoft/Detours or https://www.microsoft.com/en-us/download/details.aspx?id=52586
    	b. Unzip and go to Detours folder
    	c. set DETOURS_TARGET_PROCESSOR=X64
    	d. "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
    	e. NMAKE
    	f. copy detours.h and detours.lib in sample folder.
    1.4 Run make. For linux the dynamic library libpc_sampling_continuous.so and for windows pc_sampling_continuous.lib should be created in the folder.

2. Run the sample
    2.1 While running on linux make sure that the libpc_sampling_continuous.so, libpcsamplingutil.so and CUPTI library paths are in the LD_LIBRARY_PATH. You can do this using the command 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/CUPTI/library:<path of libpc_sampling_continuous.so>'.
    2.2 While running on windows make sure that the pc_sampling_continuous.lib, pcsamplingutil.lib and CUPTI library paths are in the PATH.
    2.3 Script libpc_sampling_continuous.pl is provided to help run the application with different PC sampling options.
    2.4 Use command './libpc_sampling_continuous.pl --help' to list all the options.
    2.5 Application code need not to be modified.
