#ifndef _MANDELBROT_KERNEL_h_
#define _MANDELBROT_KERNEL_h_

#include <vector_types.h>

extern "C" void RenderImage(uchar4 *dst, const int imageW, const int imageH);
#endif
