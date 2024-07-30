#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>
#include <cmath>

class Matrix
{
public:

	static enum init
	{
		Random, Normalize, Xavier, He
	};

	Matrix();
	Matrix(int rows, int columns);
	Matrix(int rows, int columns, init initType);
	Matrix(int rows, int columns, float value);
	Matrix(const std::vector<std::vector<float>>& matrix);

	std::vector<float> Column(int index) const;
	std::vector<float> Row(int index) const;

	void SetColumn(int index, const std::vector<float>& column);
	void SetColumn(int index, const std::vector<int>& column);
	void SetRow(int index, const std::vector<float>& row);
	void SetRow(int index, const std::vector<int>& row);

	Matrix SegmentR(int startRow, int endRow) const;
	Matrix SegmentR(int startRow) const;

	Matrix SegmentC(int startColumn, int endColumn) const;
	Matrix SegmentC(int startColumn) const;

	std::vector<float> ColumnSums() const;
	std::vector<float> RowSums() const;

	// "Advanced" Math
	Matrix ExtractFeatures(int fourier, int taylor, int chebyshev, int legendre, int laguerre, float lowerNormal, float upperNormal) const;

	Matrix Normalized(float lowerRange, float upperRange) const;

	Matrix FourierSeries(int order) const;
	Matrix TaylorSeries(int order) const;
	Matrix ChebyshevSeries(int order) const;
	Matrix LegendreSeries(int order) const;
	Matrix LaguerreSeries(int order) const;

	Matrix DotProduct(const Matrix& element) const;

	std::vector<float> LogSumExp() const;

	// Basic Math
	Matrix Negative() const;
	Matrix Abs() const;

	Matrix Add(float scalar) const;
	Matrix Add(const std::vector<float>& scalar) const;
	Matrix Add(const Matrix& element) const;

	Matrix Subtract(float scalar) const;
	Matrix Subtract(const std::vector<float>& scalar) const;
	Matrix Subtract(const Matrix& element) const;

	Matrix Multiply(float scalar) const;
	Matrix Multiply(const std::vector<float>& scalar) const;
	Matrix Multiply(const Matrix& element) const;

	Matrix Divide(float scalar) const;
	Matrix Divide(const std::vector<float>& scalar) const;
	Matrix Divide(const Matrix& element) const;

	Matrix Pow(float scalar) const;
	Matrix Pow(const std::vector<float>& scalar) const;
	Matrix Pow(const Matrix& element) const;

	Matrix Cos() const;
	Matrix Sin() const;
	Matrix Acos() const;
	Matrix Asin() const;

	Matrix Exp(float base = std::exp(1.0)) const;
	Matrix Exp(const std::vector<float>& base) const;
	Matrix Exp(const Matrix& base) const;

	Matrix Log(float base = std::exp(1.0)) const;

	// Activation Functions
	Matrix Sigmoid();
	Matrix ReLU();
	Matrix LeakyReLU(float alpha = 0.1f);
	Matrix _LeakyReLU();
	Matrix ELU(float alpha = 1.0f);
	Matrix _ELU();
	Matrix Tanh();
	Matrix Softplus();
	Matrix SiLU();

	Matrix SoftMax();

	// Activation Derivatives
	Matrix SigmoidDerivative();
	Matrix ReLUDerivative();
	Matrix LeakyReLUDerivative(float alpha = 0.1f);
	Matrix _LeakyReLUDerivative();
	Matrix ELUDerivative(float alpha = 1.0f);
	Matrix _ELUDerivative();
	Matrix TanhDerivative();
	Matrix SoftplusDerivative();
	Matrix SiLUDerivative();

	Matrix Transpose();

	Matrix Combine(Matrix element);
	Matrix Join(Matrix element);

	void Insert(int startRow, Matrix element);

	std::string ToString();
	std::string Size();

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
		Matrix mat = this->Add(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator += (std::vector<float> scalar) {
		Matrix mat = this->Add(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator += (Matrix element) {
		Matrix mat = this->Add(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}

	Matrix operator -= (float scalar) {
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator -= (std::vector<float> scalar) {
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator -= (Matrix element) {
		Matrix mat = this->Subtract(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}

	Matrix operator *= (float scalar) {
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator *= (std::vector<float> scalar) {
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator *= (Matrix element) {
		Matrix mat = this->Multiply(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}

	Matrix operator /= (float scalar) {
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator /= (std::vector<float> scalar) {
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}
	Matrix operator /= (Matrix element) {
		Matrix mat = this->Divide(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		return matrix;
	}

	std::vector<std::vector<float>> matrix;

	bool transposeBuilt = false;

private:

	std::vector<std::vector<float>> matrixT;

	Matrix SingleFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, float scalar) const;
	Matrix VectorFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const std::vector<float>& scalar) const;
	Matrix MatrixFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo) const,
		float (Matrix::* remainderOperation)(float a, float b) const, const Matrix& element) const;

	__m256 SIMDAdd(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDSub(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDMul(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDDiv(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDPow(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDExp(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDMax(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDAbs(__m256 opOne, __m256 opTwo) const;

	// SIMD Trig
	__m256 SIMDSin(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDCos(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDSec(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDCsc(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDAcos(__m256 opOne, __m256 opTwo) const;
	__m256 SIMDAsin(__m256 opOne, __m256 opTwo) const;

	float RemainderAdd(float a, float b) const;
	float RemainderSub(float a, float b) const;
	float RemainderMul(float a, float b) const;
	float RemainderDiv(float a, float b) const;
	float RemainderPow(float a, float b) const;
	float RemainderExp(float a, float b) const;
	float RemainderMax(float a, float b) const;
	float RemainderAbs(float a, float b) const;

	// SIMD Trig
	float RemainderSin(float a, float b) const;
	float RemainderCos(float a, float b) const;
	float RemainderSec(float a, float b) const;
	float RemainderCsc(float a, float b) const;
	float RemainderAcos(float a, float b) const;
	float RemainderAsin(float a, float b) const;
};