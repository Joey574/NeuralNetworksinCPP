#include <cuda_runtime.h>

__global__ void add_scalar(float* matrix, float scalar, int num_elements);
__global__ void add_vector(float* matrix, float* vec, int num_elements);

__global__ void sub_scalar(float* matrix, float scalar, int num_elements);
__global__ void sub_vector(float* matrix, float* vec, int num_elements);

__global__ void mul_scalar(float* matrix, float scalar, int num_elements);
__global__ void mul_vector(float* matrix, float* vec, int num_elements);

__global__ void div_scalar(float* matrix, float scalar, int num_elements);
__global__ void div_vector(float* matrix, float* vec, int num_elements);

__global__ void pow_scalar(float* matrix, float scalar, int num_elements);
__global__ void pow_vector(float* matrix, float* vec, int num_elements);

__global__ void sqrt_scalar(float* matrix, float scalar, int num_elements);
__global__ void sqrt_vector(float* matrix, float* vec, int num_elements);

