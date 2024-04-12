#pragma once
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <random>
#include <immintrin.h> 
#include <string>
#include "MatrixKernals.cuh"

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
	Matrix(int rows, std::vector<float> value);
	Matrix(std::vector<std::vector<float>>);

	std::string ToString();

	std::vector<float> FlattenMatrix();
	Matrix Transpose();

	void SetColumn(int index, std::vector<float> column);
	void SetColumn(int index, std::vector<int> column);
	void SetRow(int index, std::vector<float> row);
	void SetRow(int index, std::vector<int> row);

	std::vector<float> Column(int index);
	std::vector<float> Row(int index);

	std::vector<float> ColumnSums();
	std::vector<float> RowSums();

	Matrix SegmentR(int startRow, int endRow);
	Matrix SegmentR(int startRow);

	Matrix SegmentC(int startColumn, int endColumn);
	Matrix SegmentC(int startColumn);

	void Insert(int startRow, Matrix element);
	Matrix Combine(Matrix element);

	// Advanced math
	Matrix NormalizeTo(float lowerRange, float upperRange);

	Matrix FourierSeries(int order);
	Matrix TaylorSeries(int order);

	Matrix DotProduct(Matrix element);

	std::vector<float> LogSumExp();

	// Basic Math
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

	Matrix Exp(float base = std::exp(1.0));
	Matrix Exp(std::vector<float> base);
	Matrix Exp(Matrix base);

	Matrix Sqrt();

	Matrix Sin();
	Matrix Cos();

	// Activation functions and derivatives
	Matrix ReLU();
	Matrix ReLUDeriv();

	Matrix LeakyReLU(float alpha = 0.1f);
	Matrix LeakyReLUDeriv(float alpha = 0.1f);

	Matrix ELU(float alpha = 1.0f);
	Matrix ELUDeriv(float alpha = 1.0f);

	Matrix Tanh();
	Matrix TanhDeriv();

	Matrix Sigmoid();
	Matrix SigmoidDeriv();

	Matrix Softplus();
	Matrix SoftplusDeriv();

	Matrix SiLU();
	Matrix SiLUDerivative();

	Matrix Swish();
	Matrix SwishDerivative();


	// Other
	int ColumnCount;
	int RowCount;

	std::vector<std::vector<float>> matrix;

	bool transposeBuilt;
	bool flattenBuilt;


	// Operators
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
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator += (std::vector<float> scalar) {
		Matrix mat = this->Add(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator += (Matrix element) {
		Matrix mat = this->Add(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}

	Matrix operator -= (float scalar) {
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator -= (std::vector<float> scalar) {
		Matrix mat = this->Subtract(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator -= (Matrix element) {
		Matrix mat = this->Subtract(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}

	Matrix operator *= (float scalar) {
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator *= (std::vector<float> scalar) {
		Matrix mat = this->Multiply(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator *= (Matrix element) {
		Matrix mat = this->Multiply(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}

	Matrix operator /= (float scalar) {
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator /= (std::vector<float> scalar) {
		Matrix mat = this->Divide(scalar);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}
	Matrix operator /= (Matrix element) {
		Matrix mat = this->Divide(element);
		matrix = mat.matrix;
		transposeBuilt = false;
		flattenBuilt = false;
		return matrix;
	}

private:

	std::vector<std::vector<float>> matrixT;
	std::vector<float> matrixF;

	std::vector<std::vector<float>> ReshapeMatrix(std::vector<float> mat);

	Matrix SingleFloatOperation(float scalar, void (*operation)(float* matrix, float scalar, int num_elements));
	Matrix VectorFloatOperation(std::vector<float> scalar, void (*operation)(float* matrix, float* element, int num_elements));
	Matrix MatrixFloatOperation(Matrix element, void (*operation)(float* matrix, float* element, int num_elements));
};