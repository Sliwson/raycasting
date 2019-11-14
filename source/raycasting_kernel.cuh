#include <stdio.h>
#include "helper_cuda.h"
#include "raycasting_kernel.h"

// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
} 
