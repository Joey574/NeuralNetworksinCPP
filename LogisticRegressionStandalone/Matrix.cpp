#include "Matrix.h"

// Constructors
Matrix::Matrix() 
{
	matrix = std::vector<std::vector<float>>(0);
}

Matrix::Matrix(int rows, int columns) 
{
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, float value)
{
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns, value);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, float lowerRand, float upperRand)
{
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	for (int r = 0; r < matrix.size(); r++) {
		for (int c = 0; c < matrix[0].size(); c++) {
			matrix[r][c] = lowerRand + ((float)std::rand() / upperRand) * (upperRand - lowerRand);
		}
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	this->matrix = matrix;
	ColumnCount = matrix.size();
	RowCount = matrix[0].size();
}

// Util

std::vector<float> Matrix::ColumnSums() {
	std::vector<float> sums = std::vector<float>(ColumnCount);

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {
			sums[c] += matrix[r][c];
		}
	}


	// TODO: Implement parallel sum loop
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [sums, matrix](auto&& item) {
		int index = std::distance(matrix.begin(), &item)
		sums[index] = matrix[index].accumulate(item.begin(), item.end());
		});*/

	return sums;
}

// Math Operations

Matrix Matrix::Add(float scalar) {

	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [](auto&& value) {
			return value += scalar;
			});
		});*/

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar](auto&& value) {
			return value += scalar;
			});
		});

	return matrix;
}