#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"

// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
} 

// Geometry
__device__ inline void Clamp(float& value, float min = 0, float max = 1)
{
	if (value < min) value = min;
	if (value > max) value = max;
}

__device__ inline void ClampColor(float3& color)
{
	Clamp(color.x);
	Clamp(color.y);
	Clamp(color.z);
}

__device__ inline float3 Add(const float3& lhs, const float3& rhs)
{
	return {
		lhs.x + rhs.x,
		lhs.y + rhs.y,
		lhs.z + rhs.z
	};
}
__device__ inline float3 Subtract(const float3& lhs, const float3& rhs)
{
	return {
		lhs.x - rhs.x,
		lhs.y - rhs.y,
		lhs.z - rhs.z,
	};
}

__device__ inline float3 Multiply(const float3& lhs, const float scalar)
{
	return {
		lhs.x * scalar,
		lhs.y * scalar,
		lhs.z * scalar
	};
}

__device__ inline void TranslatePoint(float3& p, float* M)
{
	p.x = p.x * M[0] + p.y * M[4] + p.z * M[8] + M[12];
	p.y = p.x * M[1] + p.y * M[5] + p.z * M[9] + M[13];
	p.z = p.x * M[2] + p.y * M[6] + p.z * M[10] + M[14];

	float w = p.x * M[3] + p.y * M[7] + p.z * M[11] + M[15];
	if (w != 1 && w != 0)
		p.x /= w; p.y /= w; p.z /= w;
}

__device__ inline void TranslateVector(float3& v, float* M)
{
	v.x = v.x * M[0] + v.y * M[4] + v.z * M[8];
	v.y = v.x * M[1] + v.y * M[5] + v.z * M[9];
	v.z = v.x * M[2] + v.y * M[6] + v.z * M[10];
}

__device__ inline float LengthSquared(float3& vector)
{
	return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
}

__device__ inline float Length(float3& vector)
{
	return sqrt(LengthSquared(vector));
}

__device__ inline void Normalize(float3& vector)
{
	float length = Length(vector);
	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
}

__device__ inline float Dot(float3& v1, float3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ inline bool Intersect(float3& sphereCenter, float radius, float3& rayOrigin, float3& rayDirection, float3& intersection)
{
	float3 L = Subtract(sphereCenter, rayOrigin);
	float tCa = Dot(L, rayDirection);

	if (tCa < 0)
		return false;

	float d2 = Dot(L, L) - tCa * tCa;
	float R2 = radius * radius;
	if (d2 > R2)
		return false;

	float thc = sqrt(R2 - d2);
	float t0 = tCa - thc;
	float t1 = tCa + thc;

	if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
	if (t0 < 0)
	{
		t0 = t1;
		if (t0 < 0)
			return false;
	}

	intersection = Add(rayOrigin, Multiply(rayDirection, t0));
	return true;
}
