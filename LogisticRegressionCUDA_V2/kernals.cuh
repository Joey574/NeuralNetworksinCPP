#include <cuda_runtime.h>

__global__ void forward(float* weight, float* bias, float* input, float* total, float* activation);