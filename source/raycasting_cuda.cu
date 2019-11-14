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
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int pixel = y * imageW + x;

	if (x < imageW && y < imageH)
	{
		dst[pixel].x = (int)((float)x / imageW * 255);
		dst[pixel].y = (int)((float)y / imageH * 255);
		dst[pixel].z = 0;
	}
} 


void RenderScene(uchar4 *dst, const int imageW, const int imageH)
{
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));

	Render<<<grid, threads>>>(dst, imageW, imageH);

    getLastCudaError("Raycasting kernel execution failed.\n");
}
