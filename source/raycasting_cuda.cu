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
constexpr int maxSpheres = 2048;
constexpr int maxLights = 128;

__constant__ Sphere spheres[maxSpheres];
__constant__ float3 lights[maxLights];

float3 lightsHost[maxLights];
Sphere spheresHost[maxSpheres];
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

	if (x < imageW && y < imageH)
	{
		auto color = GetColorOpt(imageW, imageH, x, y, sphereCount, lightCount, gameTimer);
		ClampColor(color);
		dst[pixel].x = color.x * 255;
		dst[pixel].y = color.y * 255;
		dst[pixel].z = color.z * 255;
	}
} 

void ManipulateSpheres(Sphere* spheresH, const int imageW, const int imageH, const int sphereCount, float timer)
{
	//earth
	spheresH[0] = { { 0, 10.3f, -4.f }, 10.f, { .0f, .9f, .4f } };
		
	// Lissajous curve
	const float A = 5.f;
	const float B = 4.f;
	const float sigma = PI / 4.f;
	const float interval = .2f;
	timer /= 4.f;
	
	const auto x = [timer, A, sigma, sphereCount, interval](int i) {
		return sin(A * timer + sigma + i * interval) * 0.6f * (1.f + (float)log(i));
	};

	const auto y = [timer, B, sigma, sphereCount, interval](int i) {
		return (cos(B * timer + i * interval) * 0.25f - 0.25f) * (1 + (float)log(i));
	};

	const int predefinedColors = 4;
	const float3 colors[predefinedColors] = {
		{ .9f, .9f, .9f },
		{ .2f, .4f, .9f },
		{ .2f, .9f, .4f },
		{ .3f, .2f, .6f } };

	for (int i = 1; i < sphereCount; i++)
		spheresH[i] = { { x(i), y(i), -2.f - 0.3f * i }, 0.1f, colors[i % predefinedColors] };
}

void ManipulateLights(float3* lightsH, const int imageW, const int imageH, const int lightCount, float timer)
{
	const float s = timer / 3.f;
	lightsH[0] = { cos(s) * 6.f, -3.f, sin(s) * 6.f };
	lightsH[1] = { cos(s + PI / 2.f) * 6.f, -3.f, sin(s + PI / 2.f) * 6.f };
}

void RenderScene(uchar4 *dst, const int imageW, const int imageH, float gameTimer)
{
	//create spheres and lights, copy them to gpu
	const int sphereCount = 1024;
	ManipulateSpheres(spheresHost, imageW, imageH, sphereCount, gameTimer);
	
	const int lightCount = 2;
	ManipulateLights(lightsHost, imageW, imageH, lightCount, gameTimer);

	cudaMemcpyToSymbol(spheres, spheresHost, sizeof(Sphere) * sphereCount);
	cudaMemcpyToSymbol(lights, lightsHost, sizeof(float3) * lightCount);

	//execute kernel
    dim3 threads(blockX, blockY);
    dim3 grid(iDivUp(imageW, blockX), iDivUp(imageH, blockY));
	Render<<<grid, threads>>>(dst, imageW, imageH, sphereCount, lightCount, gameTimer);
    getLastCudaError("Raycasting kernel execution failed.\n");
}
