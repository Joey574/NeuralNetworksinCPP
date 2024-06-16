#include <iostream>
#include <chrono>
#include <windows.h>

#include "TestMatrix.h"

int iterations = 100;

int main()
{
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	Matrix a = Matrix(2, 3);
	Matrix b = Matrix(3, 2);

	a.SetRow(0, std::vector<float>{1, 2, 3});
	a.SetRow(1, std::vector<float>{4, 5, 6});

	b.SetColumn(0, std::vector<float>{7, 9, 11});
	b.SetColumn(1, std::vector<float>{8, 10, 12});

	std::cout << a.ToString() << std::endl;
	std::cout << b.ToString() << std::endl;

	std::cout << a.DotProductM(b).ToString() << std::endl;

	std::cout << a.DotProduct(b).ToString() << std::endl;

	return 0;

	// TEST CASE
	auto s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {

	}
	auto e = std::chrono::high_resolution_clock::now();
	auto t1 = e - s;
	std::cout << t1.count() << "ns :: " << t1.count() / 1000000.00 << "ms" << std::endl;


	// CONTROL
	s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {

	}
	e = std::chrono::high_resolution_clock::now();
	auto t2 = e - s;
	std::cout << t2.count() << "ns :: " << t2.count() / 1000000.00 << "ms" << std::endl;

	float ratio = ((t2.count() / 1000000.00) / (t1.count() / 1000000.00));
	std::cout << "Ratio: " << ratio << " :: " << (ratio > 1 ? "Test is faster" : "Control is faster") << std::endl;
}