#include <stdio.h>
#include "Geometry.h"
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

	Matrix4<float> cameraToWorld;
	float scale = tan(PI / 4);
	float imageAspectRatio = imageW / (float)imageH;
	
	auto origin = cameraToWorld.Translate(Point3<float>());

	float rayx = (2 * ((float)x + 0.5f) / (float)imageW - 1) * scale;
	float rayy = (1 - 2 * ((float)y + 0.5) / (float)imageH) * scale / imageAspectRatio;
	auto direction = cameraToWorld.Translate(Vector3<float>(rayx, rayy, -1));
	direction.Normalize();

	Point3<float> p1, p2;
	auto sphere = Sphere<float>(Point3<float>(0, 0, -100), 20.f);
	auto result = sphere.Intersect(Ray<float>(origin, direction), &p1, &p2);

	if (x < imageW && y < imageH)
	{
		if (result)
		{
			dst[pixel].x = (int)((float)x / imageW * 255);
			dst[pixel].y = (int)((float)y / imageH * 255);
			dst[pixel].z = 0;
		}
		else
		{
			dst[pixel].x = dst[pixel].y = dst[pixel].z = 0;
		}
	}
} 


void RenderScene(uchar4 *dst, const int imageW, const int imageH)
{
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));

	Render<<<grid, threads>>>(dst, imageW, imageH);

    getLastCudaError("Raycasting kernel execution failed.\n");
}
