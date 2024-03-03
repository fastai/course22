#!/usr/bin/perl

use strict;
use Getopt::Long;
use POSIX;
use Cwd qw(cwd);

my $applicationName;
my $injectionParameters;
my $cmdLineOptions;

my $verbose;
my $help;

my $injectionLibrary;

my $collectionMode;
my $samplingPeriod;
my $scratchBufferSize;
my $hwBufferSize;

my $pcConfigBufRecordCount;
my $circularBufferSize;
my $circularBufferCount;
my $fileName;
my $disableFileDump;

# Command line arguments
GetOptions( 'help'                             => \$help
          , 'app=s'                            => \$applicationName
          , 'collection-mode=i'                => \$collectionMode
          , 'sampling-period=i'                => \$samplingPeriod
          , 'scratch-buf-size=i'               => \$scratchBufferSize
          , 'hw-buf-size=i'                    => \$hwBufferSize
          , 'pc-config-buf-record-count=i'     => \$pcConfigBufRecordCount
          , 'pc-circular-buf-record-count=i'   => \$circularBufferSize
          , 'circular-buf-count=i'             => \$circularBufferCount
          , 'disable-file-dump'                => \$disableFileDump
          , 'file-name=s'                      => \$fileName
          , 'verbose'                          => \$verbose
          ) or printUsage();

# Parse and validate command line arguments
{
    if ($help) {
        printUsage();
        exit 0
    }

    if (!$applicationName) {
        printUsage();
        exit -1;
    }

    if ($collectionMode) {
        if (!($collectionMode == 1 || $collectionMode == 2))
        {
            print "ERROR : Wrong argument to --collection-mode. \n";
            printUsage();
        }
        $cmdLineOptions .= " --collection-mode ".$collectionMode;
    }

    if ($samplingPeriod) {
        if (!($samplingPeriod >= 5 && $samplingPeriod <= 31))
        {
            print "ERROR : Wrong argument to --sampling-period.\n";
            printUsage();
        }
        $cmdLineOptions .= " --sampling-period ".$samplingPeriod;
    }

    if ($scratchBufferSize) {
        $cmdLineOptions .= " --scratch-buf-size ".$scratchBufferSize;
    }

    if ($hwBufferSize) {
        $cmdLineOptions .= " --hw-buf-size ".$hwBufferSize;
    }

    if ($pcConfigBufRecordCount) {
        $cmdLineOptions .= " --pc-config-buf-record-count ".$pcConfigBufRecordCount;
    }

    if ($circularBufferSize) {
        $cmdLineOptions .= " --pc-circular-buf-record-count ".$circularBufferSize;
    }

    if ($circularBufferCount) {
        $cmdLineOptions .= " --circular-buf-count ".$circularBufferCount;
    }

    if ($fileName) {
        $cmdLineOptions .= " --file-name ".$fileName;
    }

    if ($disableFileDump) {
        $cmdLineOptions .= " --disable-file-dump ";
    }

    if ($verbose) {
        $cmdLineOptions .= " --verbose ";
    }
}

init();
RunApplication();

# End of Perl Script
# Functions definitions below

# Usage
sub printUsage {
    print STDERR "Usage: libpc_sampling_continuous.pl <options> --app <app>\n";
    print STDERR "Options:\n";
    print STDERR "  --help                          : Print help\n";
    print STDERR "  --app                           : Application to profile. Provide it in double quotes\n";
    print STDERR "  --collection-mode               : 1 - CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS
                                    2 - CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED
                                    Default : 1 \n";
    print STDERR "  --sampling-period               : Sampling period [5-31]
                                    This will set the sampling period to (2^samplingperiod) cycles\n";
    print STDERR "  --scratch-buf-size              : Scratch buffer size in bytes
                                    DEFAULT - 1 MB, which can accommodate approximately 5500 PCs
                                    with all stall reasons
                                    Approximately it takes 16 Bytes (and some fixed size memory)
                                    to accommodate one PC with one stall reason
                                    For e.g. 1 PC with 1 stall reason = 32 Bytes
                                                1 PC with 2 stall reason = 48 Bytes
                                                1 PC with 4 stall reason = 96 Bytes\n";
    print STDERR "  --hw-buf-size                   : Hardware buffer size in bytes
                                    DEFAULT - 512 MB\n";
    print STDERR "  --pc-config-buf-record-count    : PC record count for buffer used for pc sampling configuration.
                                    DEFAULT : 5000\n";
    print STDERR "  --pc-circular-buf-record-count  : PC record count in single circular buffer
                                    used to get records from CUPTI periodically or after each range.
                                    DEFAULT : 500\n";
    print STDERR "  --circular-buf-count            : Number of buffer in circular buffer.
                                    DEFAULT : 10\n";
    print STDERR "  --disable-file-dump             : Disable dumping pc sampling data in the file.
                                    DEFAULT : file dump is enabled\n";
    print STDERR "  --file-name                     : File name to store PC sampling data.
                                    DEFAULT : pcsampling.dat\n";
    print STDERR "  --verbose                       : Verbose output\n";

    print STDERR "\nExample : ./libpc_sampling_continuous.pl --collection-mode 1 --sampling-period 7 --file-name pcsampling.dat --app \"a.out --args\" \n";
    exit
}

sub init {
    my $ldLibraryPath;
    my @libPaths;

    if($^O =~ /MSWin32/) {
        $ldLibraryPath = $ENV{'PATH'};
        @libPaths = split /;/, $ldLibraryPath;
        my $dir = cwd;
        push(@libPaths, $dir);
    }
    else {
        $ldLibraryPath = $ENV{'LD_LIBRARY_PATH'};
        @libPaths = split /:/, $ldLibraryPath;
    }
    my $injectionLibraryPresent = 0;
    my $cuptiLibraryPresent = 0;
    my $utilLibraryPresent = 0;

    if(@libPaths) {
        foreach my $path (@libPaths) {
            opendir(DIR, $path);
            if(grep(/pc_sampling_continuous/, readdir(DIR))) {
                $injectionLibraryPresent = 1;
            }
            closedir(DIR);

            opendir(DIR, $path);
            if(grep(/cupti/, readdir(DIR))) {
                $cuptiLibraryPresent = 1;
            }
            closedir(DIR);

            opendir(DIR, $path);
            if(grep(/pcsamplingutil/, readdir(DIR))) {
                $utilLibraryPresent = 1;
            }
            closedir(DIR);
        }
    }

    # Set injection path
    $ENV{CUDA_INJECTION64_PATH} = ($^O =~ /MSWin32/) ? "pc_sampling_continuous.dll" : "libpc_sampling_continuous.so";

    if ($verbose) {
        if($^O =~ /linux/) {
            print "\n*** LD_LIBRARY_PATH : " . $ENV{'LD_LIBRARY_PATH'} . "\n";
        }
        print "*** CUDA_INJECTION64_PATH : " . $ENV{'CUDA_INJECTION64_PATH'} . "\n\n";
    }

    if(!($injectionLibraryPresent && $cuptiLibraryPresent && $utilLibraryPresent)) {
        if($^O =~ /MSWin32/) {
            if (!$injectionLibraryPresent) {
                print "===== ERROR: Library pc_sampling_continuous.dll not present in any of the library paths.\n";
            }
            if (!$cuptiLibraryPresent) {
                print "===== ERROR: Library cupti.dll not present in any of the library paths.\n";
            }
            if (!$utilLibraryPresent) {
                print "===== ERROR: Library pcsamplingutil.dll not present in any of the library paths.\n";
            }
            print "\nPATH : " . $ENV{'PATH'} . "\n\n";
            print "==== NOTE: Paths to pc_sampling_continuous.dll, cupti.dll and pcsamplingutil.dll libraries should be set in PATH.\n";
        }
        else {
            if (!$injectionLibraryPresent) {
                print "===== ERROR: Library libpc_sampling_continuous.so not present in any of the library paths.\n";
            }
            if (!$cuptiLibraryPresent) {
                print "===== ERROR: Library libcupti.so not present in any of the library paths.\n";
            }
            if (!$utilLibraryPresent) {
                print "===== ERROR: Library libpcsamplingutil.so not present in any of the library paths.\n";
            }
            print "\nLD_LIBRARY_PATH : " . $ENV{'LD_LIBRARY_PATH'} . "\n\n";
            print "==== NOTE: Paths to libpc_sampling_continuous.so, libcupti.so and libpcsamplingutil.so libraries should be set in LD_LIBRARY_PATH.\n";
        }
        exit(1);
    }
}

sub RunApplication {
    $injectionParameters = "$cmdLineOptions";
    $ENV{INJECTION_PARAM} = $injectionParameters;

    my $returnCode = system($applicationName);

    if($returnCode != 0) {
        print "&&&& Failed with exit code : $?\n";
    }
}
