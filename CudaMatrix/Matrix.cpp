#include "Matrix.h"

Matrix::Matrix() {
	transposeBuilt = false;
	flattenBuilt = false;
	matrix = std::vector<std::vector<float>>(0);
	ColumnCount = 0;
	RowCount = 0;
}

Matrix::Matrix(int rows, int columns) {
	transposeBuilt = false;
	flattenBuilt = false;

	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, float value) {
	transposeBuilt = false;
	flattenBuilt = false;

	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns, value);
	}

	ColumnCount = columns;
	RowCount = rows;
}

Matrix::Matrix(int rows, int columns, init initType) {
	transposeBuilt = false;
	flattenBuilt = false;

	matrix = std::vector<std::vector<float>>(rows);

	float lowerRand = -0.5;
	float upperRand = 0.5;

	for (int i = 0; i < rows; i++) {
		matrix[i] = std::vector<float>(columns);
	}

	ColumnCount = columns;
	RowCount = rows;

	std::random_device rd;
	std::mt19937 gen(rd());

	if (initType == Matrix::init::Xavier) {
		lowerRand = -(1.0f / std::sqrt(RowCount));
		upperRand = 1.0f / std::sqrt(RowCount);;

		std::uniform_real_distribution<float> dist(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen);
			}
		}
	}
	else if (initType == Matrix::init::He) {
		std::normal_distribution<float> dist(0.0, std::sqrt(2.0f / RowCount));

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen);
			}
		}
	}
	else if (initType == Matrix::init::Normalize) {
		std::uniform_real_distribution<float> dist(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen);
			}
		}
	}
	else if (initType == Matrix::init::Random) {
		std::uniform_real_distribution<float> dist(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen) * std::sqrt(1.0f / ColumnCount);
			}
		}
	}
}

Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	transposeBuilt = false;
	flattenBuilt = false;
	this->matrix = matrix;
	ColumnCount = matrix[0].size();
	RowCount = matrix.size();
}


std::string Matrix::ToString() {
	std::string out = "";

	for (int r = 0; r < RowCount; r++) {
		for (int c = 0; c < ColumnCount; c++) {
			out += std::to_string(matrix[r][c]) + " ";
		}
		out += "\n";
	}

	return out;
}


std::vector<float> Matrix::FlattenMatrix() {
	if (flattenBuilt) {
		return matrixF;
	}
	else {
		matrixF = std::vector<float>(RowCount * ColumnCount);
		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrixF[r * ColumnCount + c] = matrix[r][c];
			}
		}
		flattenBuilt = true;
		return matrixF;
	}
}

std::vector<std::vector<float>> Matrix::ReshapeMatrix(std::vector<float> mat) {

	std::vector<std::vector<float>> reshape;

	for (int r = 0; r < RowCount; r++) {
		std::vector<float> col = std::vector<float>(ColumnCount);
		for (int c = 0; c < ColumnCount; c++) {
			col[c] = mat[r * ColumnCount + c];
		}
		reshape.push_back(col);
	}

	return reshape;
}

Matrix Matrix::Transpose() {
	if (transposeBuilt) {
		return matrixT;
	}
	else {
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

Matrix Matrix::Add(float scalar) {
	return SingleFloatOperation(scalar, &add_scalar);
}
Matrix Matrix::Add(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &add_vector);
}
Matrix Matrix::Add(Matrix element) {
	return MatrixFloatOperation(element, &add_vector);
}

Matrix Matrix::Subtract(float scalar) {
	return SingleFloatOperation(scalar, &sub_scalar);
}
Matrix Matrix::Subtract(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &sub_vector);
}
Matrix Matrix::Subtract(Matrix element) {
	return MatrixFloatOperation(element, &sub_vector);
}

Matrix Matrix::Multiply(float scalar) {
	return SingleFloatOperation(scalar, &mul_scalar);
}
Matrix Matrix::Multiply(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &mul_vector);
}
Matrix Matrix::Multiply(Matrix element) {
	return MatrixFloatOperation(element, &mul_vector);
}

Matrix Matrix::Divide(float scalar) {
	return SingleFloatOperation(scalar, &div_scalar);
}
Matrix Matrix::Divide(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &div_vector);
}
Matrix Matrix::Divide(Matrix element) {
	return MatrixFloatOperation(element, &div_vector);
}

Matrix Matrix::Pow(float scalar) {
	return SingleFloatOperation(scalar, &pow_scalar);
}
Matrix Matrix::Pow(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &pow_vector);
}
Matrix Matrix::Pow(Matrix element) {
	return MatrixFloatOperation(element, &pow_vector);
}

Matrix Matrix::Exp(float scalar) {
	return SingleFloatOperation(scalar, &exp_scalar);
}
Matrix Matrix::Exp(std::vector<float> scalar) {
	return VectorFloatOperation(scalar, &exp_vector);
}
Matrix Matrix::Exp(Matrix element) {
	return MatrixFloatOperation(element, &exp_vector);
}


Matrix Matrix::SingleFloatOperation(float scalar, void (*operation)(float* matrix, float scalar, int num_elements)) {

	 std::vector<float> flat = FlattenMatrix();

	// Allocate memory for the flattened matrix on the device
	float* d_matrix;
	cudaMalloc(&d_matrix, flat.size() * sizeof(float));

	// Transfer data from host to device
	cudaMemcpy(d_matrix, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	int thr_per_blk = 256;
	int blk_in_grid = (flat.size() + thr_per_blk - 1) / thr_per_blk;

	// Launch kernel
	operation <<<blk_in_grid, thr_per_blk>>> (d_matrix, scalar, flat.size());

	// Copy the updated data back to the host
	cudaMemcpy(flat.data(), d_matrix, flat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(d_matrix);

	return ReshapeMatrix(flat);
}

Matrix Matrix::VectorFloatOperation(std::vector<float> scalar, void(*operation)(float* matrix, float* element, int num_elements) {

}

Matrix Matrix::MatrixFloatOperation(Matrix element, void (*operation)(float* matrix, float* element, int num_elements)) {
	std::vector<float> flatMat = FlattenMatrix();
	std::vector<float> elementMat = element.FlattenMatrix();

	// Allocate memory for the flattened matrix on the device
	float* d_matrix;
	float* e_matrix;

	cudaMalloc(&d_matrix, flatMat.size() * sizeof(float));
	cudaMalloc(&e_matrix, elementMat.size() * sizeof(float));

	// Transfer data from host to device
	cudaMemcpy(d_matrix, flatMat.data(), flatMat.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(e_matrix, elementMat.data(), elementMat.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	int thr_per_blk = 256;
	int blk_in_grid = (flatMat.size() + thr_per_blk - 1) / thr_per_blk;

	// Launch kernel
	operation <<<blk_in_grid, thr_per_blk>>> (d_matrix, e_matrix, flatMat.size());

	// Copy the updated data back to the host
	cudaMemcpy(flatMat.data(), d_matrix, flatMat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(d_matrix);
	cudaFree(e_matrix);

	return ReshapeMatrix(flatMat);
}

