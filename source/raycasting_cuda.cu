#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"
#include "raycasting_kernel.cuh"

namespace {
constexpr int blockX = 16;
constexpr int blockY = 16;
}

__global__ void Render(uchar4 *dst, const int imageW, const int imageH)
{
	
} 


void RenderImage(uchar4 *dst, const int imageW, const int imageH)
{
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));

	Render<<<threads, grid>>>(dst, imageW, imageH);

    getLastCudaError("Raycasting kernel execution failed.\n");
}
