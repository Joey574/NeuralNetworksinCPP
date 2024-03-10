#include <iostream>

#include "Matrix.h"

using namespace std;

int main()
{
	Matrix test = Matrix(10, 100, 1);

	cout << "Colc: " << test.ColumnCount << endl;
	cout << "Rowc:" << test.RowCount << endl;

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;

	test.Add(45);

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
}
