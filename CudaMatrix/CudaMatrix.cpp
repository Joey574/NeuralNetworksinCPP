#include <iostream>

#include "Matrix.h"

int main()
{
	Matrix a = Matrix(10, 5, 4);
	Matrix b = Matrix(10, 5, 3);

	std::vector<float> c = { 1, 2, 3, 4, 5 };
	std::vector<float> d = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

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
	std::cout << "A ^ B:\n" << a.Pow(b).ToString() << std::endl << std::endl;

	// Vector float tests
	std::cout << "Vector Float Tests:\n\n";
	std::cout << "A + C:\n" << a.Add(c).ToString() << std::endl;
	std::cout << "A - C:\n" << a.Subtract(c).ToString() << std::endl;
	std::cout << "A * C:\n" << a.Multiply(c).ToString() << std::endl;
	std::cout << "A / C:\n" << a.Divide(c).ToString() << std::endl;
	std::cout << "A ^ C:\n" << a.Pow(c).ToString() << std::endl << std::endl;

	std::cout << "B + D:\n" << b.Add(d).ToString() << std::endl;
	std::cout << "B - D:\n" << b.Subtract(d).ToString() << std::endl;
	std::cout << "B * D:\n" << b.Multiply(d).ToString() << std::endl;
	std::cout << "B / D:\n" << b.Divide(d).ToString() << std::endl;
	std::cout << "B ^ D:\n" << b.Pow(d).ToString() << std::endl << std::endl;
}
