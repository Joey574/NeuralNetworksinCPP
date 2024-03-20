#include <iostream>

#include "Matrix.h"

using namespace std;

bool AddTest();
bool SubTest();
bool MulTest();
bool DivTest();
bool PowTest();
bool ExpTest();

int main()
{
	cout << "Begin Matrix Class Test:" << endl;

	bool add = AddTest();

	if (add) {
		cout << "\tAddition Tests Succesful..." << endl;
	}
	else {
		cout << "\tAddition Tests Failed..." << endl;
	}

	bool sub = AddTest();

	if (sub) {
		cout << "\tSubtraction Tests Succesful..." << endl;
	}
	else {
		cout << "\tSubtraction Tests Failed..." << endl;
	}

	bool mul = AddTest();

	if (mul) {
		cout << "\tMultiplication Tests Succesful..." << endl;
	}
	else {
		cout << "\tMultiplication Tests Failed..." << endl;
	}

	bool div = AddTest();

	if (div) {
		cout << "\tDivision Tests Succesful..." << endl;
	}
	else {
		cout << "\tDivision Tests Failed..." << endl;
	}

	bool pow = AddTest();

	if (pow) {
		cout << "\tPower Tests Succesful..." << endl;
	}
	else {
		cout << "\tPower Tests Failed..." << endl;
	}

	bool exp = AddTest();

	if (exp) {
		cout << "\tExponential Tests Succesful..." << endl;
	}
	else {
		cout << "\tExponential Tests Failed..." << endl;
	}

	if (add && sub && mul && div && pow && exp) {
		cout << "All Tests Succesful..." << endl;
	}
	else {
		cout << "Test Failure..." << endl;
	}

}

bool AddTest() {

	bool matAdd = true;
	Matrix a = Matrix(10, 10, 5);
	int expectedValue = a[0][0] + a[0][0];

	Matrix b = a + a;

	for (int r = 0; r < b.RowCount; r++) {
		for (int c = 0; c < b.ColumnCount; c++) {
			if (b[r][c] != expectedValue) {
				matAdd = false;
				break;
			}
		}
	}

	bool vecAdd = true;
	vector<float> vec = vector<float>(10, 1);
	expectedValue = a[0][0] + vec[0];

	b = a + vec;

	for (int r = 0; r < b.RowCount; r++) {
		for (int c = 0; c < b.ColumnCount; c++) {
			if (b[r][c] != expectedValue) {
				vecAdd = false;
				break;
			}
		}
	}

	bool floatAdd = true;
	float f = 15;
	expectedValue = a[0][0] + f;
	b = a + f;

	for (int r = 0; r < b.RowCount; r++) {
		for (int c = 0; c < b.ColumnCount; c++) {
			if (b[r][c] != expectedValue) {
				floatAdd = false;
				break;
			}
		}
	}

	return matAdd && vecAdd && floatAdd;
}

bool SubTest() {
	return 0;
}

bool MulTest() {
	return 0;
}

bool DivTest() {
	return 0;
}

bool PowTest() {
	return 0;
}

bool ExpTest() {
	return 0;
}