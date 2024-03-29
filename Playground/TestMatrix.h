#pragma once
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>

class Matrix
{
public:

	Matrix();
	Matrix(int rows, int columns);
	Matrix(int rows, int columns, float value);
	Matrix(int rows, int columns, float lowerRand, float upperRand);
	Matrix(std::vector<std::vector<float>>);

	std::vector<float> Column(int index);
	std::vector<float> Row(int index);

	void SetColumn(int index, std::vector<float> column);
	void SetColumn(int index, std::vector<int> column);
	void SetRow(int index, std::vector<float> row);
	void SetRow(int index, std::vector<int> row);

	std::vector<float> ColumnSums();
	std::vector<float> RowSums();
	std::vector<float> RowSumsSeq();

	Matrix Add(float scalar);
	Matrix Add(std::vector<float> scalar);
	Matrix Add(Matrix element);

	Matrix Subtract(float scalar);
	Matrix Subtract(std::vector<float> scalar);
	Matrix Subtract(Matrix element);

	Matrix Multiply(float scalar);
	Matrix Multiply(std::vector<float> scalar);
	Matrix Multiply(Matrix element);

	Matrix Divide(float scalar);
	Matrix Divide(std::vector<float> scalar);
	Matrix Divide(Matrix element);

	Matrix Pow(float scalar);
	Matrix Pow(std::vector<float> scalar);
	Matrix Pow(Matrix element);

	Matrix Exp();
	Matrix Exp(float base);
	Matrix Exp(std::vector<float> base);
	Matrix Exp(Matrix base);

	std::vector<float> LogSumExp();

	Matrix Transpose();

	Matrix DotProduct(Matrix element);

	std::string ToString();

	int ColumnCount;
	int RowCount;

	std::vector<float>& operator[] (int index) {
		return matrix[index];
	}

	Matrix operator + (float scalar) {
		return this->Add(scalar);
	}
	Matrix operator + (std::vector<float> scalar) {
		return this->Add(scalar);
	}
	Matrix operator + (Matrix element) {
		return this->Add(element);
	}

	Matrix operator - (float scalar) {
		return this->Subtract(scalar);
	}
	Matrix operator - (std::vector<float> scalar) {
		return this->Subtract(scalar);
	}
	Matrix operator - (Matrix element) {
		return this->Subtract(element);
	}

	Matrix operator * (float scalar) {
		return this->Multiply(scalar);
	}
	Matrix operator * (std::vector<float> scalar) {
		return this->Multiply(scalar);
	}
	Matrix operator * (Matrix element) {
		return this->Multiply(element);
	}

	Matrix operator / (float scalar) {
		return this->Divide(scalar);
	}
	Matrix operator / (std::vector<float> scalar) {
		return this->Divide(scalar);
	}
	Matrix operator / (Matrix element) {
		return this->Divide(element);
	}


	Matrix operator += (float scalar) {
		transposeBuilt = false;
		Matrix mat = this->Add(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator += (std::vector<float> scalar) {
		transposeBuilt = false;
		Matrix mat = this->Add(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator += (Matrix element) {
		transposeBuilt = false;
		Matrix mat = this->Add(element);
		matrix = mat.matrix;
		return matrix;
	}

	Matrix operator -= (float scalar) {
		transposeBuilt = false;
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator -= (std::vector<float> scalar) {
		transposeBuilt = false;
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator -= (Matrix element) {
		transposeBuilt = false;
		Matrix mat = this->Subtract(element);
		matrix = mat.matrix;
		return matrix;
	}

	Matrix operator *= (float scalar) {
		transposeBuilt = false;
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator *= (std::vector<float> scalar) {
		transposeBuilt = false;
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator *= (Matrix element) {
		transposeBuilt = false;
		Matrix mat = this->Multiply(element);
		matrix = mat.matrix;
		return matrix;
	}

	Matrix operator /= (float scalar) {
		transposeBuilt = false;
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator /= (std::vector<float> scalar) {
		transposeBuilt = false;
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		return matrix;
	}
	Matrix operator /= (Matrix element) {
		transposeBuilt = false;
		Matrix mat = this->Divide(element);
		matrix = mat.matrix;
		return matrix;
	}

	std::vector<std::vector<float>> matrix;

	bool transposeBuilt = false;

private:

	std::vector<std::vector<float>> matrixT;

	Matrix SingleFloatOperation(void (Matrix::* operation)(__m256 opOne, __m256 opTwo, __m256* result),
		float (Matrix::* remainderOperation)(float a, float b), float scalar);
	Matrix VectorFloatOperation(void (Matrix::* operation)(__m256 opOne, __m256 opTwo, __m256* result),
		float (Matrix::* remainderOperation)(float a, float b), std::vector<float> scalar);
	Matrix MatrixFloatOperation(void (Matrix::* operation)(__m256 opOne, __m256 opTwo, __m256* result),
		float (Matrix::* remainderOperation)(float a, float b), Matrix element);

	std::vector<float> HorizontalSum(std::vector<std::vector<float>> element);
	std::vector<float> VerticalSum(std::vector<std::vector<float>> element);

	void SIMDAdd(__m256 opOne, __m256 opTwo, __m256* result);
	void SIMDSub(__m256 opOne, __m256 opTwo, __m256* result);
	void SIMDMul(__m256 opOne, __m256 opTwo, __m256* result);
	void SIMDDiv(__m256 opOne, __m256 opTwo, __m256* result);
	void SIMDPow(__m256 opOne, __m256 opTwo, __m256* result);
	void SIMDExp(__m256 opOne, __m256 opTwo, __m256* result);

	float RemainderAdd(float a, float b);
	float RemainderSub(float a, float b);
	float RemainderMul(float a, float b);
	float RemainderDiv(float a, float b);
	float RemainderPow(float a, float b);
	float RemainderExp(float a, float b);
};