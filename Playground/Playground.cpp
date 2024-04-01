#include <iostream>
#include <chrono>
#include <windows.h>

#include "TestMatrix.h"

int iterations = 100;

Matrix Sigmoid(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = 1 / (1 + std::exp(-total[r][c]));
		}
	}
	return a;
}

Matrix SigmoidM(Matrix total) {
	Matrix a = Matrix(total.RowCount, total.ColumnCount, 1.0f);
	return  a / (total.Negative().Exp() + 1);
}

int main()
{
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	Matrix a = Matrix(8, 80000, -0.5f, 0.5f);
	Matrix b = a;

	// TEST CASE
	auto s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		SigmoidM(a);
	}
	auto e = std::chrono::high_resolution_clock::now();
	auto t1 = e - s;
	std::cout << t1.count() << "ns :: " << t1.count() / 1000000.00 << "ms" << std::endl;


	// CONTROL
	s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		Sigmoid(a);
	}
	e = std::chrono::high_resolution_clock::now();
	auto t2 = e - s;
	std::cout << t2.count() << "ns :: " << t2.count() / 1000000.00 << "ms" << std::endl;

	float ratio = ((t2.count() / 1000000.00) / (t1.count() / 1000000.00));
	std::cout << "Ratio: " << ratio << " :: " << (ratio > 1 ? "Test is faster" : "Control is faster") << std::endl;
}