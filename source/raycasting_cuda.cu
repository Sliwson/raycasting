#include <stdio.h>
#include "Geometry.h"
#include "helper_cuda.h"
#include "raycasting_kernel.h"
#include "raycasting_kernel.cuh"

namespace {
constexpr int blockX = 16;
constexpr int blockY = 16;
}

__device__ float3 GetColor(const int imageW, const int imageH, const int x, const int y)
{
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
	auto light = Point3<float>(-200, -200, -200);
	float3 color = { 110.f / 255, 193.f / 255, 248.f / 255 };
	auto sphere = Sphere<float>(Point3<float>(0, imageH / 10, -4), imageH / 10);
	float3 sphereColor = { .9f, .9f, 0.f };
	float kd = 0.5;
	float ks = 0.5;
	int alpha = 10;

	//intersection
	Point3<float> intersection;
	auto result = sphere.Intersect(cameraRay, &intersection);
	if (result)
	{
		auto normal = Vector3<float>(intersection - sphere.C);
		normal.Normalize();
		auto lightVector = Vector3<float>(light - intersection);
		lightVector.Normalize();

		auto r = normal * 2.f * Vector3<float>::Dot(lightVector, normal) - lightVector;
		auto view = Vector3<float>(intersection - origin);
		view.Normalize();

		float kdm = kd * Vector3<float>::Dot(normal, lightVector);
		float ksm = ks * pow(Vector3<float>::Dot(r, view), alpha);
		float multiplier = kdm + ksm;

		color.x = sphereColor.x * multiplier;
		color.y = sphereColor.y * multiplier;
		color.z = sphereColor.z * multiplier;
	}

	return color;
}

__device__ float3 GetColorOpt(const int imageW, const int imageH, const int x, const int y, float gameTimer)
{
	float cameraToWorld[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	float scale = tan(PI / 4);
	float imageAspectRatio = imageW / (float)imageH;

	//getcamera ray
	float3 cameraOrigin = { 0, 0, 0 };
	TranslatePoint(cameraOrigin, cameraToWorld);

	//get camera ray
	float rayx = (2 * ((float)x + 0.5f) / (float)imageW - 1) * scale * imageAspectRatio;
	float rayy = (1 - 2 * ((float)y + 0.5) / (float)imageH) * scale;
	float3 cameraRayDirection = { rayx, rayy, -1 };
	TranslatePoint(cameraRayDirection, cameraToWorld);
	Normalize(cameraRayDirection);

	//hardcoded constants
	const int lightCount = 3;
	float s = gameTimer / 2;
	float3 lightPositions[lightCount] = { {0, sinf(s) * imageW, cosf(s) * imageW},
										{sinf(s + PI * 0.66) * imageW, -200, cosf(s + PI * 0.66) * imageW}, 
										{sinf(s + 1.33 * PI) * imageW, -200, cosf(s + 1.33 * PI) * 300 - 300} };

	float3 outColor = { 110.f / 255, 193.f / 255, 248.f / 255 };
	
	float3 sphereCenter = { 0, imageH / 11, -4 };
	float sphereRadius = imageH / 11;
	float3 sphereColor = { .9f, .9f, 0.f };
	
	float kd = 0.5;
	float ks = 0.5;
	int alpha = 10;

	//intersection
	float3 intersection = { 0, 0, 0 };
	bool result = Intersect(sphereCenter, sphereRadius, cameraOrigin, cameraRayDirection, intersection);
	if (result)
	{
		float3 normal = Subtract(intersection, sphereCenter);
		Normalize(normal);
		outColor = { 0, 0, 0 };

		for (int i = 0; i < lightCount; i++)
		{
			float3 lightVector = Subtract(lightPositions[i], intersection);
			Normalize(lightVector);

			float3 r = Subtract(Multiply(normal, 2.f * Dot(lightVector, normal)), lightVector);
			float3 viewVector = Subtract(intersection, cameraOrigin);
			Normalize(viewVector);

			float kdm = kd * Dot(normal, lightVector);
			float ksm = ks * pow(Dot(r, viewVector), alpha);
			float multiplier = kdm + ksm;
			outColor = Add(outColor, Multiply(sphereColor, multiplier));
		}
	}

	return outColor;
}

__global__ void Render(uchar4 *dst, const int imageW, const int imageH, float gameTimer)
{
	//calculate pixel coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int pixel = y * imageW + x;

	auto color = GetColorOpt(imageW, imageH, x, y, gameTimer);
	ClampColor(color);
	if (x < imageW && y < imageH)
	{
		dst[pixel].x = color.x * 255;
		dst[pixel].y = color.y * 255;
		dst[pixel].z = color.z * 255;
	}
} 


void RenderScene(uchar4 *dst, const int imageW, const int imageH, float gameTimer)
{
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));

	Render<<<grid, threads>>>(dst, imageW, imageH, gameTimer);

    getLastCudaError("Raycasting kernel execution failed.\n");
}
