#include <iostream>

#include "Matrix.h"

int main()
{
	Matrix a = Matrix(10, 10, 4);
	Matrix b = Matrix(10, 10, 3);

	// Single float tests
	/*std::cout << "Before add:\n" << a.ToString() << std::endl;
	std::cout << "After add:\n" << a.Add(5).ToString() << std::endl;

	std::cout << "Before sub:\n" << a.ToString() << std::endl;
	std::cout << "After sub:\n" << a.Subtract(14).ToString() << std::endl;

	std::cout << "Before mul:\n" << a.ToString() << std::endl;
	std::cout << "After mul:\n" << a.Multiply(2).ToString() << std::endl;

	std::cout << "Before div:\n" << a.ToString() << std::endl;
	std::cout << "After div:\n" << a.Divide(2).ToString() << std::endl;

	std::cout << "Before pow:\n" << a.ToString() << std::endl;
	std::cout << "After pow:\n" << a.Pow(3).ToString() << std::endl;*/

	// Matrix float tests
	/*std::cout << "A + B:\n" << a.Add(b).ToString() << std::endl;
	std::cout << "A - B:\n" << a.Subtract(b).ToString() << std::endl;
	std::cout << "A * B:\n" << a.Multiply(b).ToString() << std::endl;
	std::cout << "A / B:\n" << a.Divide(b).ToString() << std::endl;
	std::cout << "A ^ B:\n" << a.Pow(b).ToString() << std::endl;*/
}
