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

__global__ void exp_scalar(float* matrix, float scalar, int num_elements);
__global__ void exp_vector(float* matrix, float* vec, int num_elements);

__global__ void sqrt(float* matrix, float scalar, int num_elements);

__global__ void sin(float* matrix, float scalar, int num_elements);
__global__ void cos(float* matrix, float scalar, int num_elements);

__global__ void relu(float* matrix, float sclar, int num_elements);
__global__ void relu_derivative(float* matrix, float sclar, int num_elements);

__global__ void leakyrelu(float* matrix, float sclar, int num_elements);
__global__ void leakyrelu_derivative(float* matrix, float sclar, int num_elements);