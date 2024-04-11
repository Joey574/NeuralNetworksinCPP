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

	static enum init
	{
		Random, Normalize, Xavier, He
	};

	Matrix();
	Matrix(int rows, int columns);
	Matrix(int rows, int columns, init initType);
	Matrix(int rows, int columns, float value);
	Matrix(std::vector<std::vector<float>>);


	Matrix Add(float scalar);
	Matrix Add(std::vector<float> scalar);
	Matrix Add(Matrix element);


	int ColumnCount;
	int RowCount;


	std::vector<std::vector<float>> matrix;
};

