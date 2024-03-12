#pragma once
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 

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

		std::vector<float> SetColumn(int index, std::vector<float> column);
		std::vector<float> SetRow(int index, std::vector<float> row);
		
		std::vector<float> MultiplyAndSum(float scalar);

		std::vector<float> ColumnSums();
		std::vector<float> RowSums();
		
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

		Matrix CollapseAndLeftMultiply(Matrix element);

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

		std::vector<std::vector<float>> matrix;

	private:

		Matrix SingleFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), float scalar);
		Matrix VectorFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), std::vector<float> scalar);
		Matrix MatrixFloatOperation(void (Matrix::*operation)(__m256 opOne, __m256 opTwo, __m256* result), Matrix element);

		void SIMDAdd(__m256 opOne, __m256 opTwo, __m256* result);
		void SIMDSub(__m256 opOne, __m256 opTwo, __m256* result);
		void SIMDMul(__m256 opOne, __m256 opTwo, __m256* result);
		void SIMDDiv(__m256 opOne, __m256 opTwo, __m256* result);
		void SIMDPow(__m256 opOne, __m256 opTwo, __m256* result);
};