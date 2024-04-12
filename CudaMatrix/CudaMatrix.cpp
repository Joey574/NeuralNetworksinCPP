#include <iostream>

#include "Matrix.h"

int main()
{
	Matrix a = Matrix(10, 10, 4);
	Matrix b = Matrix(10, 10, 3);

	//Single float tests
	std::cout << "Single Float Tests:\n\n";
	std::cout << "A + 5:\n" << a.Add(5).ToString() << std::endl;
	std::cout << "A - 14:\n" << a.Subtract(14).ToString() << std::endl;
	std::cout << "A * 2:\n" << a.Multiply(2).ToString() << std::endl;
	std::cout << "A / 2:\n" << a.Divide(2).ToString() << std::endl;
	std::cout << "A ^ 3:\n" << a.Pow(3).ToString() << std::endl << std::endl;

	// Matrix float tests
	std::cout << "Matrix Float Tests:\n\n";
	std::cout << "A + B:\n" << a.Add(b).ToString() << std::endl;
	std::cout << "A - B:\n" << a.Subtract(b).ToString() << std::endl;
	std::cout << "A * B:\n" << a.Multiply(b).ToString() << std::endl;
	std::cout << "A / B:\n" << a.Divide(b).ToString() << std::endl;
	std::cout << "A ^ B:\n" << a.Pow(b).ToString() << std::endl;
}
