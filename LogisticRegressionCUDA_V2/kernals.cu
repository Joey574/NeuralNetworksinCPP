#include "kernals.cuh"

__global__ void forward(float* weight, float* bias, float* input, float* total, float* activation, int weightSize, int biasSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < weightSize) {

	}
}

//aTotal[i] = (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose();
//activation[i] = i < aTotal.size() - 1 ? (aTotal[i].LeakyReLU()) : SoftMax(aTotal[i]);
