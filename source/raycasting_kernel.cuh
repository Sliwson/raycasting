#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"

// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
} 

__device__ inline void Clamp(float* value, float min = 0, float max = 1)
{
	if (*value < min) *value = min;
	if (*value > max) *value = max;
}

__device__ inline void ClampColor(float3* color)
{
	Clamp(&color->x);
	Clamp(&color->y);
	Clamp(&color->z);
}
