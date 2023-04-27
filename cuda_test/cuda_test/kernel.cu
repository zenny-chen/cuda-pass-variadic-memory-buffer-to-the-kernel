
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>

static __global__ void memArgsKernel(int *args[2])
{
    auto const localThreadID = threadIdx.x;
    auto const blockSize = blockDim.x;
    auto const blockID = blockIdx.x;
    auto const globalThreadID = blockID * blockSize + localThreadID;

    args[0][globalThreadID] += args[1][globalThreadID];
}

static void TestMemArgs(void)
{
    int* devMem0 = nullptr;
    int* devMem1 = nullptr;
    int* *devArgMem = nullptr;

    constexpr int elemCount = 4096;
    constexpr auto bufferSize = elemCount * sizeof(*devMem0);

    int* mainMem = (int*)malloc(bufferSize);
    if (mainMem == nullptr)
        return;

    for (int i = 0; i < elemCount; i++)
        mainMem[i] = 1;

    do
    {
        auto cudaStatus = cudaMalloc(&devMem0, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&devMem1, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        int* args[] = { devMem0, devMem1 };

        cudaStatus = cudaMalloc(&devArgMem, sizeof(args));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devMem0, mainMem, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devMem1, mainMem, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devArgMem, args, sizeof(args), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaDeviceProp props{ };
        cudaStatus = cudaGetDeviceProperties(&props, 0);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        auto const threadCount = props.maxThreadsPerBlock;
        auto const blockCount = elemCount / threadCount;
        memArgsKernel <<< blockCount, threadCount >>> (devArgMem);

        cudaStatus = cudaMemcpy(mainMem, devMem0, bufferSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Verify the result
        for (int i = 0; i < elemCount; i++)
        {
            if (mainMem[i] != 2)
            {
                printf("Error occurred @ %d\n", i);
                break;
            }
        }

    } while (false);

    free(mainMem);

    if (devMem0 != nullptr)
        cudaFree(devMem0);
    if (devMem1 != nullptr)
        cudaFree(devMem1);
    if (devArgMem != nullptr)
        cudaFree(devArgMem);
}

int main(void)
{
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    TestMemArgs();

    puts("Test completed!!");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }
}

