#include "MatrixKernals.cuh"

__global__ void vecadd(float* a, float* b, float* c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	c[i] = a[i] + b[i];
}
