#include <iostream>
#include <chrono>
#include <windows.h>

#include "TestMatrix.h"

int iterations = 5000;

int main()
{
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	Matrix a = Matrix(1, 8000, 1);
	Matrix b = Matrix(a.RowCount, a.ColumnCount, 5);

	// TEST CASE
	auto s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		a.RowSums();
	}
	auto e = std::chrono::high_resolution_clock::now();
	auto t1 = e - s;
	std::cout << t1.count() << "ns :: " << t1.count() / 1000000.00 << "ms" << std::endl;


	// SEQUENTIAL
	s = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		b.RowSumsSeq();
	}
	e = std::chrono::high_resolution_clock::now();
	auto t2 = e - s;
	std::cout << t2.count() << "ns :: " << t2.count() / 1000000.00 << "ms" << std::endl;


	std::cout << "Ratio: " << ((t2.count() / 1000000.00) / (t1.count() / 1000000.00)) << std::endl;
}