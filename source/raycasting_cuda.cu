#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"
#include "raycasting_kernel.cuh"

namespace {
//constants
constexpr int blockX = 16;
constexpr int blockY = 16;
constexpr auto PI = 3.14159265358979323846f;

//spheres
constexpr int maxSpheres = 1024;
constexpr int maxLights = 128;

__constant__ Sphere spheres[maxSpheres];
__constant__ float3 lights[maxLights];
}

// device optimized code
__device__ float3 GetColorOpt(const int imageW, const int imageH, const int x, const int y, const int sphereCount, const int lightCount, float gameTimer)
{
	float cameraToWorld[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	float scale = tan(PI / 4);
	float imageAspectRatio = imageW / (float)imageH;

	//getcamera ray
	float3 cameraOrigin = { 0, 0, 0 };
	TranslatePoint(cameraOrigin, cameraToWorld);

	float rayx = (2 * ((float)x + 0.5f) / (float)imageW - 1) * scale * imageAspectRatio;
	float rayy = (1 - 2 * ((float)y + 0.5) / (float)imageH) * scale;
	float3 cameraRayDirection = { rayx, rayy, -4 };
	TranslatePoint(cameraRayDirection, cameraToWorld);
	Normalize(cameraRayDirection);

	float3 outColor = { 110.f / 255, 193.f / 255, 248.f / 255 };
	
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
			float3 lightVector = Subtract(lights[i], intersection);
			Normalize(lightVector);

			float3 r = Subtract(Multiply(normal, 2.f * Dot(lightVector, normal)), lightVector);
			float3 viewVector = Subtract(cameraOrigin, intersection);
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

__global__ void Render(uchar4 *dst, const int imageW, const int imageH, const int sphereCount, const int lightCount, float gameTimer)
{
	//calculate pixel coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int pixel = y * imageW + x;

	auto color = GetColorOpt(imageW, imageH, x, y, sphereCount, lightCount, gameTimer);
	ClampColor(color);
	if (x < imageW && y < imageH)
	{
		dst[pixel].x = color.x * 255;
		dst[pixel].y = color.y * 255;
		dst[pixel].z = color.z * 255;
	}
} 

void ManipulateSpheres(Sphere* spheresH, const int imageW, const int imageH, const int sphereCount, float timer)
{
	spheresH[0] = { { 0, imageH / 11.f, -4 }, imageH / 11.f, { .0f, .9f, 0.4f } };
	spheresH[1] = { { -0.4, -0.4, -3 }, 0.1f, { .9f, .9f, 9.f } };
	spheresH[2] = { { 0.4, -0.4, -3}, 0.1f, { 0, .4f, .9f } };
	spheresH[3] = { { -0.6, -0.6, -3}, 0.1f, { 0, .9f, .4f } };
	spheresH[4] = { { 0.6, -0.6, -3}, 0.1f, { .3f, .2f, .6f } };
}

void ManipulateLights(float3* lightsH, const int imageW, const int imageH, const int lightCount, float timer)
{
	const float s = timer / 2.f;
	lightsH[0] = { 0, sinf(s) * imageW * 2, cosf(s) * imageW * 2 };
	lightsH[1] = { sinf(s + PI * 0.66f) * imageW * 2, -200, cosf(s + PI * 0.66f) * imageW * 2 }; 
}

void RenderScene(uchar4 *dst, const int imageW, const int imageH, float gameTimer)
{
	//create spheres and lights, copy them to gpu
	const int sphereCount = 5;
	Sphere spheresHost[sphereCount];
	ManipulateSpheres(spheresHost, imageW, imageH, sphereCount, gameTimer);
	
	const int lightCount = 2;
	float3 lightsHost[lightCount];
	ManipulateLights(lightsHost, imageW, imageH, lightCount, gameTimer);

	cudaMemcpyToSymbol(spheres, spheresHost, sizeof(Sphere) * sphereCount);
	cudaMemcpyToSymbol(lights, lightsHost, sizeof(float3) * lightCount);

	//execute kernel
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));
	Render<<<grid, threads>>>(dst, imageW, imageH, sphereCount, lightCount, gameTimer);
    getLastCudaError("Raycasting kernel execution failed.\n");
}
