#include "cuda_utils.h"
#include <stdio.h>

void cuda_device_info() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("CUDA Device Information:\n");
    printf("  Device Count: %d\n", deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("  Device 0: %s\n", prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Total Global Memory: %.2f GB\n",
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("    Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    }
    printf("\n");
}
