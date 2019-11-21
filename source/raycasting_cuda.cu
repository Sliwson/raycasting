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
	//calculate pixel coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int pixel = y * imageW + x;

	//calculate camera ray
	Matrix4<float> cameraToWorld;
	float scale = tan(PI / 4);
	float imageAspectRatio = imageW / (float)imageH;
	auto origin = cameraToWorld.Translate(Point3<float>());
	float rayx = (2 * ((float)x + 0.5f) / (float)imageW - 1) * scale * imageAspectRatio;
	float rayy = (1 - 2 * ((float)y + 0.5) / (float)imageH) * scale;
	auto direction = cameraToWorld.Translate(Vector3<float>(rayx, rayy, -1));
	direction.Normalize();
	auto cameraRay = Ray<float>(origin, direction);

	float3 color = { 110.f / 255, 193.f / 255, 248.f / 255 };
	
	Point3<float> p1, p2;
	auto sphere = Sphere<float>(Point3<float>(-30, -30, -100), 20.f);
	auto result = sphere.Intersect(cameraRay, &p1, &p2);
	if (result)
	{
		auto l1 = Vector3<float>(p1 - origin).LengthSquared();
		auto l2 = Vector3<float>(p2 - origin).LengthSquared();

		if (l2 < l1)
			p1 = p2;

		auto normal = Vector3<float>(p1 - sphere.C);
		color = { 1, 1, 1 };
	}

	ClampColor(&color);
	if (x < imageW && y < imageH)
	{
		dst[pixel].x = color.x * 255;
		dst[pixel].y = color.y * 255;
		dst[pixel].z = color.z * 255;
	}
} 


void RenderScene(uchar4 *dst, const int imageW, const int imageH)
{
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));

	Render<<<grid, threads>>>(dst, imageW, imageH);

    getLastCudaError("Raycasting kernel execution failed.\n");
}
