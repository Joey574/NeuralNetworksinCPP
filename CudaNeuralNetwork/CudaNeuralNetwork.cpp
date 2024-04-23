#include <iostream>
#include <cuda_runtime.h>

__global__ void train_netowrk() {

}

__device__ void forward_propogation(float* weights, float* biases, float* input, float* total, float* activation, float* dimensions) {

	// aTotal[i] = weights[i].DotProd(i == 0 in : activation[i - 1]) + biases[i]
	// activation[i] = i < aTotal.size() - 1 ? leakyrelu(aTotal[i]) : softmax(aTotal[i])
}

__device__ void backward_propogation() {

}

__device__ void leakyrelu(float* matrix, float scalar) {

}

__device__ void leakyrelu_derivative(float* matrix, float scalar) {

}


int main()
{

}
