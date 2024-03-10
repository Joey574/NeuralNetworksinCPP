#pragma once
#include <vector>
#include <algorithm>
#include <execution>

class Matrix {
	public:

		Matrix();
		Matrix(int rows, int columns);
		Matrix(int rows, int columns, float value);
		Matrix(int rows, int columns, float lowerRand, float upperRand);
		Matrix(std::vector<std::vector<float>>);

		//std::vector<float> Column(int index);
		//std::vector<float> Row(int index);
		//
		std::vector<float> ColumnSums();
		//std::vector<float> RowSums();
		//
		Matrix Add(float scalar);
		//Matrix Add(std::vector<float> scalar);
		//Matrix Add(Matrix scalar);
		//
		//Matrix Subtract(float scalar);
		//Matrix Subtract(std::vector<float> scalar);
		//Matrix Subtract(Matrix scalar);
		//
		//Matrix Multiply(float scalar);
		//Matrix Multiply(std::vector<float> scalar);
		//Matrix Multiply(Matrix scalar);
		//
		//Matrix Divide(float scalar);
		//Matrix Divide(std::vector<float> scalar);
		//Matrix Divide(Matrix scalar);

		int ColumnCount;
		int RowCount;

	private:

		std::vector<std::vector<float>> matrix;
};