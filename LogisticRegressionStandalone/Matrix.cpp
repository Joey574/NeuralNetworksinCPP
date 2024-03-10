#include "Matrix.h"

// Constructors
Matrix::Matrix() {
	matrix = std::vector<std::vector<float>>(0);
}

Matrix::Matrix(int rows, int columns) {
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, float value) {
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns, value);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, float lowerRand, float upperRand) {
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	for (int r = 0; r < matrix.size(); r++) {
		for (int c = 0; c < matrix[0].size(); c++) {
			matrix[r][c] = 0;
		}
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	this->matrix = matrix;
	ColumnCount = matrix[0].size();
	RowCount = matrix.size();
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

std::vector<float> Matrix::RowSums() {
	std::vector<float> sums = std::vector<float>(RowCount);

	for (int r = 0; r < RowCount; r++) {
		sums[r] = std::reduce(matrix[r].begin(), matrix[r].end());
	}

	return sums;
}

// Math Operations

Matrix Matrix::Add(float scalar) {
	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar](auto&& value) {
			return value += scalar;
			});
		});

	return matrix;
}

Matrix Matrix::Add(std::vector<float> scalar) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] += scalar[r];
		}
	}	
 
	// TODO: Finish parallel vector-wise add
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		return std::transform(item.begin(), item.end(), scalar.begin(), item.begin(), std::plus<float>());
		});*/

	return matrix;
}

Matrix Matrix::Add(Matrix element) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] += element[r][c];
		}
	}

	// TODO: Implement parallel element-wise add with another matrix
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		int r = std::distance(matrix.begin(), &item);
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar[r]](auto&& value) {
			int c = std::distance(item.begin(), &value);
			return value += scalar[c];
			});
		});*/

	return matrix;
}


Matrix Matrix::Subtract(float scalar) {
	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar](auto&& value) {
			return value -= scalar;
			});
		});

	return matrix;
}

Matrix Matrix::Subtract(std::vector<float> scalar) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] -= scalar[r];
		}
	}

	// TODO: Finish parallel vector-wise add
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		return std::transform(item.begin(), item.end(), scalar.begin(), item.begin(), std::plus<float>());
		});*/

	return matrix;
}

Matrix Matrix::Subtract(Matrix element) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] -= element[r][c];
		}
	}

	// TODO: Implement parallel element-wise add with another matrix
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		int r = std::distance(matrix.begin(), &item);
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar[r]](auto&& value) {
			int c = std::distance(item.begin(), &value);
			return value += scalar[c];
			});
		});*/

	return matrix;
}


Matrix Matrix::Multiply(float scalar) {
	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar](auto&& value) {
			return value *= scalar;
			});
		});

	return matrix;
}

Matrix Matrix::Multiply(std::vector<float> scalar) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] *= scalar[r];
		}
	}

	// TODO: Finish parallel vector-wise add
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		return std::transform(item.begin(), item.end(), scalar.begin(), item.begin(), std::plus<float>());
		});*/

	return matrix;
}

Matrix Matrix::Multiply(Matrix element) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] *= element[r][c];
		}
	}

	// TODO: Implement parallel element-wise add with another matrix
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		int r = std::distance(matrix.begin(), &item);
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar[r]](auto&& value) {
			int c = std::distance(item.begin(), &value);
			return value += scalar[c];
			});
		});*/

	return matrix;
}


Matrix Matrix::Divide(float scalar) {
	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar](auto&& value) {
			return value /= scalar;
			});
		});

	return matrix;
}

Matrix Matrix::Divide(std::vector<float> scalar) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] /= scalar[r];
		}
	}

	// TODO: Finish parallel vector-wise add
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		return std::transform(item.begin(), item.end(), scalar.begin(), item.begin(), std::plus<float>());
		});*/

	return matrix;
}

Matrix Matrix::Divide(Matrix element) {

	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {

			matrix[r][c] /= element[r][c];
		}
	}

	// TODO: Implement parallel element-wise add with another matrix
	/*std::for_each(std::execution::par, matrix.begin(), matrix.end(), [scalar](auto&& item) {
		int r = std::distance(matrix.begin(), &item);
		std::for_each(std::execution::par, item.begin(), item.end(), [scalar[r]](auto&& value) {
			int c = std::distance(item.begin(), &value);
			return value += scalar[c];
			});
		});*/

	return matrix;
}