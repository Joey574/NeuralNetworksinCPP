#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")


#include <iostream>
#include <chrono>

#include "Matrix.h"
#include "MatrixKernals.cuh"

int main()
{
	/*Matrix a = Matrix(10, 10, 5);
	Matrix b = Matrix(10, 10, 2);

	std::cout << a.ToString() << std::endl;*/

	int num = std::pow(1024, 3);

	float* a_h = new float[num];
	float* b_h = new float[num];
	float* c_h = new float[num];

	for (int i = 0; i < num; i++) {
		a_h[i] = static_cast<float>(i);
		b_h[i] = static_cast<float>(i);
	}

    auto start = std::chrono::high_resolution_clock::now();

    float* A_device;
    float* B_device;
    float* C_device;
    cudaMalloc(&A_device, num * sizeof(float));
    cudaMalloc(&B_device, num * sizeof(float));
    cudaMalloc(&C_device, num * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(A_device, a_h, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, b_h, num * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 512;
    int gridSize = (num + blockSize - 1) / blockSize;

    // Launch the kernel
    vecadd<<<gridSize, blockSize>>>(A_device, B_device, C_device);

    // Copy the result back to the host
    cudaMemcpy(c_h, C_device, num * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;

    std::cout << "CUDA Time: " << time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num; i++) {
        c_h[i] = a_h[i] + b_h[i];
    }

    time = std::chrono::high_resolution_clock::now() - start;

    std::cout << "SEQ Time: " << time.count() << " ms";
}