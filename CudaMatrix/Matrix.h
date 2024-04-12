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
	Matrix(std::vector<std::vector<float>>);

	std::string ToString();

	std::vector<float> FlattenMatrix();

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
	Matrix Pow(std::vector<int> scalar);
	Matrix Pow(Matrix element);

	Matrix Exp(float base = std::exp(1.0));
	Matrix Exp(std::vector<float> base);
	Matrix Exp(Matrix base);


	int ColumnCount;
	int RowCount;

	std::vector<std::vector<float>> matrix;

private:

	std::vector<std::vector<float>> ReshapeMatrix(std::vector<float> mat);

	Matrix SingleFloatOperation(float scalar, void (*operation)(float* matrix, float scalar, int num_elements));
	Matrix VectorFloatOperation(std::vector<float> scalar);
	Matrix MatrixFloatOperation(Matrix element, void (*operation)(float* matrix, float* element, int num_elements));
};

