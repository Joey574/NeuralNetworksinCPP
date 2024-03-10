#include <iostream>

#include "Matrix.h"

using namespace std;

int main()
{
	Matrix test = Matrix(10, 100, -0.5f, 0.5f);
	vector<float> vec = vector<float>(10, 1);

	cout << "Colc: " << test.ColumnCount << endl;
	cout << "Rowc:" << test.RowCount << endl;

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
	cout << "RowSums[0]: " << test.RowSums()[0] << endl;

	cout << "add scalar of 9" << endl;
	test.Add(9);

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
	cout << "RowSums[0]: " << test.RowSums()[0] << endl;

	cout << "Vector-wise add vector of 1's" << endl;
	test.Add(vec);

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
	cout << "RowSums[0]: " << test.RowSums()[0] << endl;

	cout << "Element-wise add same matrix" << endl;
	test.Add(test);

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
	cout << "RowSums[0]: " << test.RowSums()[0] << endl;

	cout << "Divide by scalar 1.23" << endl;
	test.Divide(1.23f);

	cout << "ColumnSums[0]: " << test.ColumnSums()[0] << endl;
	cout << "RowSums[0]: " << test.RowSums()[0] << endl;
}
