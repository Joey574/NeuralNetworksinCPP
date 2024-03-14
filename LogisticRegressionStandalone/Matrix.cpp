#include "Matrix.h"

#include <iostream>

// Constructors

Matrix::Matrix() {
	matrix = std::vector<std::vector<float>>(0);
	ColumnCount = 0;
	RowCount = 0;
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

	ColumnCount = columns;
	RowCount = rows;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(lowerRand, upperRand);

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			matrix[r][c] = dist(gen);
		}
	}
}

Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	this->matrix = matrix;
	ColumnCount = matrix[0].size();
	RowCount = matrix.size();
}

// Util

std::vector<float> Matrix::SetColumn(int index, std::vector<float> vector) {
	return matrix[0];
}

std::vector<float> Matrix::SetColumn(int index, std::vector<int> vector) {
	std::vector<float> replaced = Column(index);

	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	return replaced;
}

std::vector<float> Matrix::SetRow(int index, std::vector<float> vector) {
	return matrix[0];
}

std::vector<float> Matrix::SetRow(int index, std::vector<int> vector) {
	return matrix[0];
}

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

std::vector<float> Matrix::Column(int index) {
	std::vector<float> column = std::vector<float>();
	for (int i = 0; i < RowCount; i++) {
		column.push_back(matrix[i][index]);
	}
	return column;
}

std::vector<float> Matrix::Row(int index) {
	return matrix[index];
}

std::vector<float> Matrix::MultiplyAndSum(float scalar) {
	std::vector<std::vector<float>> mul = matrix;
	__m256 scalarS = _mm256_set1_ps(scalar);

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&](auto&& item) {
		size_t r = &item - matrix.data();
		const int alignedN = item.size() - (item.size() % 8);
		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_loadu_ps(&item[i]);
			__m256 result = _mm256_mul_ps(loaded_a, scalarS);
			_mm256_storeu_ps(&mul[r][i], result);
		}
		});
	Matrix m = mul;

	return m.RowSums();
}

Matrix Matrix::DotProduct(Matrix element) {

	// element == input

	// input 784 x 500

	// matrix == weights

	// weights == 784 x 128

	// final == aTotal

	// 128 x 500

	std::vector<std::vector<float>> mat = std::vector<std::vector<float>>();

	for (int i = 0; i < ColumnCount; i++) {
		mat.push_back(element.Multiply(this->Column(i)).ColumnSums());
	}

	return mat;
}

Matrix Matrix::CollapseAndLeftMultiply(Matrix element) {
	std::vector<std::vector<float>> mat = matrix;
	for (int i = 0; i < element.ColumnCount; i++) {
		mat.push_back(this->Multiply(element.Column(i)).ColumnSums());
	}
	return mat;
}

void Matrix::AssignVector(Matrix element) {
	matrix = element.matrix;

	ColumnCount = element.ColumnCount;
	RowCount = element.RowCount;
}

// Math Operations

Matrix Matrix::Add(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDAdd, scalar);
}

Matrix Matrix::Add(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDAdd, scalar);
}

Matrix Matrix::Add(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDAdd, element);
}


Matrix Matrix::Subtract(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDSub, scalar);
}

Matrix Matrix::Subtract(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDSub, scalar);
}

Matrix Matrix::Subtract(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDSub, element);
}


Matrix Matrix::Multiply(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDMul, scalar);
}

Matrix Matrix::Multiply(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDMul, scalar);
}

Matrix Matrix::Multiply(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDMul, element);
}


Matrix Matrix::Divide(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDDiv, scalar);
}

Matrix Matrix::Divide(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDDiv, scalar);
}

Matrix Matrix::Divide(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDDiv, element);
}


Matrix Matrix::Pow(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDPow, scalar);
}

Matrix Matrix::Pow(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDPow, scalar);
}

Matrix Matrix::Pow(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDPow, element);
}


Matrix Matrix::Exp() {
	std::vector<std::vector<float>> exp = matrix;
	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			exp[r][c] = std::exp(matrix[r][c]);
		}
	}
	return exp;
}

Matrix Matrix::Transpose() {

	Matrix transpose = Matrix(ColumnCount, RowCount);

	for (int i = 0; i < RowCount; i++) {
		transpose.SetColumn(i, matrix[i]);
	}

	return transpose;
}

Matrix Matrix::SingleFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), float scalar) {
	std::vector<std::vector<float>> mat = matrix;
	__m256 _scalar = _mm256_set1_ps(scalar);

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&](auto&& item) {  

		size_t r = &item - matrix.data();
		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_loadu_ps(&item[i]);
			__m256 result;
			(this->*operation)(loaded_a, _scalar, &result);
			_mm256_storeu_ps(&mat[r][i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			mat[r][i] = item[i] + scalar;
		}
	});
	return mat;
}

Matrix Matrix::VectorFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), std::vector<float> scalar) {

	Matrix mat;

	if (scalar.size() == ColumnCount) {
		mat = matrix;
	} else if (scalar.size() == RowCount) {
		mat = this->Transpose();
	} else { 
		std::cout << "size no match :( " << RowCount << " :: " << ColumnCount << " b: " << scalar.size()  << std::endl;
	}

	std::for_each(std::execution::par, mat.matrix.begin(), mat.matrix.end(), [&](auto&& item) {

		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {

			__m256 loaded_a = _mm256_loadu_ps(&item[i]);
			__m256 loaded_b = _mm256_loadu_ps(&scalar[i]);
			__m256 result;

			(this->*operation)(loaded_a, loaded_b, &result);
			_mm256_storeu_ps(&item[i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			item[i] += scalar[i];
		}
	});

	return mat;
}

Matrix Matrix::MatrixFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), Matrix element) {
	std::vector<std::vector<float>> mat = element.matrix;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&](auto&& item) {

		size_t r = &item - matrix.data();
		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_loadu_ps(&item[i]);
			__m256 loaded_b = _mm256_loadu_ps(&mat[r][i]);
			__m256 result;

			(this->*operation)(loaded_a, loaded_b, &result);
			_mm256_storeu_ps(&mat[r][i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			mat[r][i] += item[i];
		}
	});
	return mat;
}

// SIMD Operations

void Matrix::SIMDAdd(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_add_ps(opOne, opTwo);
}
void Matrix::SIMDSub(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_div_ps(opOne, opTwo);
}
void Matrix::SIMDMul(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_mul_ps(opOne, opTwo);
}
void Matrix::SIMDDiv(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_div_ps(opOne, opTwo);
}
void Matrix::SIMDPow(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_pow_ps(opOne, opTwo);
}