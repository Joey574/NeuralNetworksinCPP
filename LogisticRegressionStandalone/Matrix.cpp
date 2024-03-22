#include "Matrix.h"

#include <iostream>


// Constructors

Matrix::Matrix() {
	matrix = std::vector<std::vector<float>>(0);
	ColumnCount = 0;
	RowCount = 0;
	transposeBuilt = false;
}

Matrix::Matrix(int rows, int columns) {
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	ColumnCount = columns;
	RowCount = rows;
	transposeBuilt = false;
}

Matrix::Matrix(int rows, int columns, float value) {
	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns, value);
	}

	ColumnCount = columns;
	RowCount = rows;
	transposeBuilt = false;
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
	transposeBuilt = false;
}

Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	this->matrix = matrix;
	ColumnCount = matrix[0].size();
	RowCount = matrix.size();
	transposeBuilt = false;
}

// Util

void Matrix::SetColumn(int index, std::vector<float> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
}

void Matrix::SetColumn(int index, std::vector<int> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
}

void Matrix::SetRow(int index, std::vector<float> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
}

void Matrix::SetRow(int index, std::vector<int> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
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

	if (transposeBuilt) {
		return matrixT[index];
	}

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

	std::vector<std::vector<float>> mat = std::vector<std::vector<float>>();

	for (int i = 0; i < ColumnCount; i++) {
		mat.push_back(element.Multiply(this->Column(i)).RowSums());
	}

	return mat;
}

bool Matrix::ContainsNaN() {
	bool hasNan = false;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&hasNan](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [&hasNan](auto&& var) {
			if (std::isnan(var)) {
				hasNan = true;
			}
		});
	});

	return hasNan;
}

bool Matrix::ContainsInf() {
	bool hasInf = false;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&hasInf](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [&hasInf](auto&& var) {
			if (std::isinf(var)) {
				hasInf = true;
			}
			});
		});

	return hasInf;
}

Matrix Matrix::ReplaceInf(float value) {
	transposeBuilt = false;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&value](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [&value](auto&& var) {
			if (std::isinf(var)) {
				var = value;
			}
		});
	});
	return matrix;
}

Matrix Matrix::ReplaceNAN(float value) {
	transposeBuilt = false;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&value](auto&& item) {
		std::for_each(std::execution::par, item.begin(), item.end(), [&value](auto&& var) {
			if (std::isnan(var)) {
				var = value;
			}
			});
		});
	return matrix;
}

// Math Operations

Matrix Matrix::Add(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}

Matrix Matrix::Add(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}

Matrix Matrix::Add(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
}


Matrix Matrix::Subtract(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}

Matrix Matrix::Subtract(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}

Matrix Matrix::Subtract(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
}


Matrix Matrix::Multiply(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}

Matrix Matrix::Multiply(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}

Matrix Matrix::Multiply(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
}


Matrix Matrix::Divide(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}

Matrix Matrix::Divide(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}

Matrix Matrix::Divide(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
}


Matrix Matrix::Pow(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}

Matrix Matrix::Pow(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}

Matrix Matrix::Pow(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, element);
}


Matrix Matrix::Exp() {
	return SingleFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, std::exp(1.0));
}

std::vector<float> Matrix::LogSumExp() {

	std::vector<float> logSum = std::vector<float>(ColumnCount);

	for (int c = 0; c < ColumnCount; c++) {

		std::vector<float> col = Column(c);

		auto maxElement = std::max_element(col.begin(), col.end());
		float max = *maxElement;
		float sum = 0;

		for (int i = 0; i < col.size(); i++) {
			sum += std::exp(col[i] - c);
		}
		logSum[c] = c + std::log(sum);
	}
	return logSum;
}


Matrix Matrix::Transpose() {

	if (transposeBuilt) {
		return matrixT;
	} else {
		matrixT = std::vector<std::vector<float>>(ColumnCount);

		for (int i = 0; i < ColumnCount; i++) {
			matrixT[i] = std::vector<float>(RowCount);
		}

		for (int i = 0; i < ColumnCount; i++) {
			matrixT[i] = this->Column(i);
		}

		transposeBuilt = true;

		return matrixT;
	}
}

std::string Matrix::AsString() {
	std::string out = "";

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			out += std::to_string(matrix[r][c]) + " ";
		}
		out += " :: " + std::to_string(this->RowSums()[r]);
		out += "\n";
	}

	return out;
}

Matrix Matrix::SingleFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), 
	float (Matrix::* remainderOperation)(float a, float b), float scalar) {

	std::vector<std::vector<float>> mat = matrix;
	__m256 _scalar = _mm256_set1_ps(scalar);

	std::for_each(std::execution::par, mat.begin(), mat.end(), [&](auto&& item) {  

		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_load_ps(&item[i]);
			__m256 result;
			(this->*operation)(loaded_a, _scalar, &result);
			_mm256_store_ps(&item[i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			item[i] = (this->*remainderOperation)(item[i], scalar);
		}
	});
	return mat;
}

Matrix Matrix::VectorFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), 
	float (Matrix::* remainderOperation)(float a, float b), std::vector<float> scalar) {

	Matrix mat;

	if (scalar.size() == ColumnCount) {
		mat = matrix;
	}
	else if (scalar.size() == RowCount) {
		mat = this->Transpose();
	}

	std::for_each(std::execution::par, mat.matrix.begin(), mat.matrix.end(), [&](auto&& item) {

		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {

			__m256 loaded_a = _mm256_load_ps(&item[i]);
			__m256 loaded_b = _mm256_load_ps(&scalar[i]);
			__m256 result;

			(this->*operation)(loaded_a, loaded_b, &result);
			_mm256_store_ps(&item[i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			item[i] = (this->*remainderOperation)(item[i], scalar[i]);
		}
	});

	return mat;
}

Matrix Matrix::MatrixFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), 
	float (Matrix::* remainderOperation)(float a, float b), Matrix element) {

	std::vector<std::vector<float>> mat = element.matrix;

	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&](auto&& item) {

		size_t r = &item - matrix.data();
		const int alignedN = item.size() - (item.size() % 8);

		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_load_ps(&item[i]);
			__m256 loaded_b = _mm256_load_ps(&mat[r][i]);
			__m256 result;

			(this->*operation)(loaded_a, loaded_b, &result);
			_mm256_store_ps(&mat[r][i], result);
		}

		for (int i = alignedN; i < item.size(); i++) {
			mat[r][i] = (this->*remainderOperation)(mat[r][i], item[i]);
		}
	});
	return mat;
}

// SIMD Operations

void Matrix::SIMDAdd(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_add_ps(opOne, opTwo);
}
void Matrix::SIMDSub(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_sub_ps(opOne, opTwo);
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
void Matrix::SIMDExp(__m256 opOne, __m256 opTwo, __m256* result) {
	*result = _mm256_pow_ps(opTwo, opOne);
}

float Matrix::RemainderAdd(float a, float b) {
	return a + b;
}
float Matrix::RemainderSub(float a, float b) {
	return a - b;
}
float Matrix::RemainderMul(float a, float b) {
	return a * b;
}
float Matrix::RemainderDiv(float a, float b) {
	return a / b;
}
float Matrix::RemainderPow(float a, float b) {
	return std::pow(a, b);
}
float Matrix::RemainderExp(float a, float b) {
	return std::pow(b, a);
}