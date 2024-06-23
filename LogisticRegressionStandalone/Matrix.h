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
	Matrix(std::vector<std::vector<float>>);

	std::vector<float> Column(int index);
	std::vector<float> Row(int index);

	void SetColumn(int index, std::vector<float> column);
	void SetColumn(int index, std::vector<int> column);
	void SetRow(int index, std::vector<float> row);
	void SetRow(int index, std::vector<int> row);

	Matrix SegmentR(int startRow, int endRow);
	Matrix SegmentR(int startRow);

	Matrix SegmentC(int startColumn, int endColumn);
	Matrix SegmentC(int startColumn);

	std::vector<float> ColumnSums();
	std::vector<float> RowSums();

	// "Advanced" Math
	Matrix ExtractFeatures(int fourier, int taylor, int chebyshev, int legendre, int laguerre, float lowerNormal, float upperNormal);

	Matrix FourierSeries(int order);
	Matrix TaylorSeries(int order);
	Matrix ChebyshevSeries(int order);
	Matrix LegendreSeries(int order);
	Matrix LaguerreSeries(int order);

	Matrix DotProduct(Matrix element);

	std::vector<float> LogSumExp();

	// Basic Math
	Matrix Negative();
	Matrix Abs();

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

	Matrix Cos();
	Matrix Sin();
	Matrix Acos();
	Matrix Asin();

	Matrix Exp(float base = std::exp(1.0));
	Matrix Exp(std::vector<float> base);
	Matrix Exp(Matrix base);

	Matrix Log(float base = std::exp(1.0));

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
	Matrix Normalized(float lowerRange, float upperRange);

	void Insert(int startRow, Matrix element);

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

	Matrix SingleFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo),
		float (Matrix::* remainderOperation)(float a, float b), float scalar);
	Matrix VectorFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo),
		float (Matrix::* remainderOperation)(float a, float b), std::vector<float> scalar);
	Matrix MatrixFloatOperation(__m256 (Matrix::*operation)(__m256 opOne, __m256 opTwo),
		float (Matrix::* remainderOperation)(float a, float b), Matrix element);

	std::vector<float> HorizontalSum(std::vector<std::vector<float>> element);
	std::vector<float> VerticalSum(std::vector<std::vector<float>> element);

	__m256 SIMDAdd(__m256 opOne, __m256 opTwo);
	__m256 SIMDSub(__m256 opOne, __m256 opTwo);
	__m256 SIMDMul(__m256 opOne, __m256 opTwo);
	__m256 SIMDDiv(__m256 opOne, __m256 opTwo);
	__m256 SIMDPow(__m256 opOne, __m256 opTwo);
	__m256 SIMDExp(__m256 opOne, __m256 opTwo);
	__m256 SIMDMax(__m256 opOne, __m256 opTwo);
	__m256 SIMDAbs(__m256 opOne, __m256 opTwo);

	// SIMD Trig
	__m256 SIMDSin(__m256 opOne, __m256 opTwo);
	__m256 SIMDCos(__m256 opOne, __m256 opTwo);
	__m256 SIMDSec(__m256 opOne, __m256 opTwo);
	__m256 SIMDCsc(__m256 opOne, __m256 opTwo);
	__m256 SIMDAcos(__m256 opOne, __m256 opTwo);
	__m256 SIMDAsin(__m256 opOne, __m256 opTwo);

	float RemainderAdd(float a, float b);
	float RemainderSub(float a, float b);
	float RemainderMul(float a, float b);
	float RemainderDiv(float a, float b);
	float RemainderPow(float a, float b);
	float RemainderExp(float a, float b);
	float RemainderMax(float a, float b);
	float RemainderAbs(float a, float b);

	// SIMD Trig
	float RemainderSin(float a, float b);
	float RemainderCos(float a, float b);
	float RemainderSec(float a, float b);
	float RemainderCsc(float a, float b);
	float RemainderAcos(float a, float b);
	float RemainderAsin(float a, float b);
};