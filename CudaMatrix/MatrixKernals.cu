#include "MatrixKernals.cuh"

// Single Scalar operation

__global__ void add_scalar(float* matrix, float scalar, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] += scalar;
    }
}

__global__ void sub_scalar(float* matrix, float scalar, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] -= scalar;
    }
}

__global__ void mul_scalar(float* matrix, float scalar, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] *= scalar;
    }
}

__global__ void div_scalar(float* matrix, float scalar, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] /= scalar;
    }
}

__global__ void pow_scalar(float* matrix, float scalar, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] = __powf(matrix[id], scalar);
    }
}

// Vector Operations

__global__ void add_vector(float* matrix, float* element, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] += element[id];
    }
}

__global__ void sub_vector(float* matrix, float* element, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] -= element[id];
    }
}

__global__ void mul_vector(float* matrix, float* element, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] *= element[id];
    }
}

__global__ void div_vector(float* matrix, float* element, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] /= element[id];
    }
}

__global__ void pow_vector(float* matrix, float* element, int num_elements) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
        matrix[id] = __powf(matrix[id], element[id]);
    }
}