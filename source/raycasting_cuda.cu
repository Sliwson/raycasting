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

	//hardcoded constants
	auto light = Point3<float>(0, 0, 0);
	float3 color = { 110.f / 255, 193.f / 255, 248.f / 255 };
	float3 sphereColor = { 1.f, 0.f, 0.f };
	float kd = 1;
	float ks = 0;
	int alpha = 10;

	//intersection
	Point3<float> intersection;
	auto sphere = Sphere<float>(Point3<float>(0, 0, -50), 20.f);
	auto result = sphere.Intersect(cameraRay, &intersection);
	if (result)
	{
		auto normal = Vector3<float>(intersection - sphere.C);
		normal.Normalize();
		auto lightVector = Vector3<float>(light - intersection);
		lightVector.Normalize();

		float kdm = kd * Vector3<float>::Dot(normal, lightVector);

		color.x = sphereColor.x * kdm;
		color.y = sphereColor.y * kdm;
		color.z = sphereColor.z * kdm;
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
