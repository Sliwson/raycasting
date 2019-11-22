#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"
#include "raycasting_kernel.cuh"

namespace {
constexpr int blockX = 16;
constexpr int blockY = 16;
constexpr auto PI = 3.14159265358979323846f;
}

// device optimized code
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
	const int lightCount = 2;
	float s = gameTimer / 2;
	float3 lightPositions[lightCount] = { {0, sinf(s) * imageW * 2, cosf(s) * imageW * 2},
										{sinf(s + PI * 0.66) * imageW * 2, -200, cosf(s + PI * 0.66) * imageW * 2} };

	float3 outColor = { 110.f / 255, 193.f / 255, 248.f / 255 };
	
	const int sphereCount = 5;
	Sphere spheres[sphereCount] = { { { 0, imageH / 11.f, -4 }, imageH / 11.f, { .14f, .296f, 0.225f } },
		{ { -1000, -300, -1000 }, 100.f, { .9f, .9f, 9.f } },
		{ { 0, -400, -1000}, 100.f, { 0, .4f, .9f } },
		{ { 800, -600, -1200}, 100.f, { 0, .9f, .4f } },
		{ { -800, -600, -1200}, 100.f, { .3f, .2f, .6f } } };
	
	float ka = 0.1;
	float kd = 0.5;
	float ks = 0.4;
	int alpha = 32;

	//found intersection with nearest sphere
	float3 intersection = { 0, 0, 0 };
	int iIntersected = -1;
	float minDistance = FLT_MAX;

	for (int s = 0; s < sphereCount; s++)
	{
		Sphere sphere = spheres[s];
		float3 tempIntersection = { 0, 0, 0 };
		bool result = Intersect(sphere.center, sphere.radius, cameraOrigin, cameraRayDirection, tempIntersection);
		if (result) 
		{
			float3 diff = Subtract(tempIntersection, cameraOrigin);
			float distance = LengthSquared(diff);
			if (distance < minDistance)
			{
				minDistance = distance;
				iIntersected = s;
				intersection = tempIntersection;
			}	
		}
	}

	//calculate color
	if (iIntersected > -1)
	{
		Sphere sphere = spheres[iIntersected];
		float3 normal = Subtract(intersection, sphere.center);
		Normalize(normal);
		outColor = { 0, 0, 0 };

		for (int i = 0; i < lightCount; i++)
		{
			float3 lightVector = Subtract(lightPositions[i], intersection);
			Normalize(lightVector);

			float3 r = Subtract(Multiply(normal, 2.f * Dot(lightVector, normal)), lightVector);
			float3 viewVector = Subtract(intersection, cameraOrigin);
			Normalize(viewVector);
			
			float d1 =  Dot(normal, lightVector);
			if (d1 < 0)
				d1 = 0;

			float kdm = kd * d1;
			
			float d2 = Dot(r, viewVector);
			if (d2 < 0)
				d2 = 0;

			float ksm = ks * pow(d2, alpha);
			float multiplier = kdm + ksm;
			outColor = Add(outColor, Multiply(sphere.color, multiplier));
			outColor = Add(outColor, Multiply(sphere.color, ka));
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
