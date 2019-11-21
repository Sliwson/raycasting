#ifndef _MANDELBROT_KERNEL_h_
#define _MANDELBROT_KERNEL_h_

#include <vector_types.h>

extern "C" void RenderScene(uchar4 *dst, const int imageW, const int imageH, float gameTimer);
#endif
