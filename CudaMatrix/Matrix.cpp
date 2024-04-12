#include "Matrix.h"

// Constructors
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

Matrix::Matrix(int rows, std::vector<float> value) {
	transposeBuilt = false;
	flattenBuilt = false;

	matrix = std::vector<std::vector<float>>(rows);

	for (int i = 0; i < rows; i++) {
		matrix[i] = value;
	}

	ColumnCount = value.size();
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

// Util
void Matrix::SetColumn(int index, std::vector<float> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
	flattenBuilt = false;
}
void Matrix::SetColumn(int index, std::vector<int> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
	flattenBuilt = false;
}

void Matrix::SetRow(int index, std::vector<float> vector) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
	flattenBuilt = false;
}
void Matrix::SetRow(int index, std::vector<int> vector) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
	flattenBuilt = false;
}

void Matrix::Insert(int startRow, Matrix element) {
	for (int i = 0; i < element.RowCount; i++) {
		this->SetRow(i + startRow, element.Row(i));
	}

	transposeBuilt = false;
	flattenBuilt = false;
}

std::vector<float> Matrix::Column(int index) {
	if (transposeBuilt) {
		return matrixT[index];
	}
	else {
		std::vector<float> column = std::vector<float>();
		for (int i = 0; i < RowCount; i++) {
			column.push_back(matrix[i][index]);
		}
		return column;
	}
}
std::vector<float> Matrix::Row(int index) {
	return matrix[index];
}

std::vector<float> Matrix::ColumnSums() {
	if (transposeBuilt) {
		std::vector<float> sums; sums.reserve(matrixT.size());

		for (int r = 0; r < matrixT.size(); r++) {
			float s = 0;
			for (int c = 0; c < matrixT[r].size(); c++) {
				s += matrixT[r][c];
			}
			sums.push_back(s);
		}
		return sums;
	} else {
		std::vector<float> sums = std::vector<float>(matrix[0].size());

		for (int c = 0; c < matrix[0].size(); c++) {
			for (int r = 0; r < matrix.size(); r++) {
				sums[c] += matrix[r][c];
			}
		}
		return sums;
	}
}
std::vector<float> Matrix::RowSums() {
	std::vector<float> sums = std::vector<float>(matrix.size());

	for (int r = 0; r < matrix.size(); r++) {
		//sums[r] = std::reduce(matrix[r].begin(), matrix[r].end());
	}

	return sums;
}

Matrix Matrix::SegmentR(int startRow, int endRow) {
	Matrix a = Matrix(endRow - startRow, ColumnCount);

	for (int i = 0; i < a.RowCount; i++) {
		a.SetRow(i, this->Row(i + startRow));
	}

	return a;
}
Matrix Matrix::SegmentR(int startRow) {
	Matrix a = Matrix(RowCount - startRow, ColumnCount);

	for (int i = 0; i < a.RowCount; i++) {
		a.SetRow(i, this->Row(i + startRow));
	}

	return a;
}

Matrix Matrix::SegmentC(int startColumn, int endColumn) {
	Matrix a = Matrix(RowCount, endColumn - startColumn);

	for (int i = 0; i < a.ColumnCount; i++) {
		a.SetColumn(i, this->Column(i + startColumn));
	}

	return a;
}
Matrix Matrix::SegmentC(int startColumn) {
	Matrix a = Matrix(RowCount, ColumnCount - startColumn);

	for (int i = 0; i < a.ColumnCount; i++) {
		a.SetColumn(i, this->Column(i + startColumn));
	}

	return a;
}

Matrix Matrix::Combine(Matrix element) {
	Matrix a = Matrix(RowCount + element.RowCount, ColumnCount);

	for (int i = 0; i < RowCount; i++) {
		a.SetRow(i, this->Row(i));
	}

	for (int i = 0; i < element.RowCount; i++) {
		a.SetRow(i + RowCount, element.Row(i));
	}

	return a;
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

// Advanced Math
Matrix Matrix::NormalizeTo(float lowerRange, float upperRange) {
	Matrix normal = matrix;

	float fMin = FLT_MAX;
	float fMax = FLT_MIN;

	for (int c = 0; c < ColumnCount; c++) {
		std::vector<float> vec = this->Column(c);

		float min = *std::min_element(vec.begin(), vec.end());
		float max = *std::max_element(vec.begin(), vec.end());

		if (min < fMin) {
			fMin = min;
		}

		if (max > fMax) {
			fMax = max;
		}
	}

	for (int c = 0; c < ColumnCount; c++) {
		std::vector<float> vec = this->Column(c);

		std::vector<float> n;

		for (float x : vec) {
			float temp = lowerRange + ((x - fMin) / (fMax - fMin)) * (upperRange - lowerRange);
			n.push_back(temp);
		}
		normal.SetColumn(c, n);
	}
	return normal;
}

Matrix Matrix::FourierSeries(int order) {
	return this->Multiply(order).SingleFloatOperation(0, &sin).Combine
	(this->Multiply(order).SingleFloatOperation(0, &cos));
}
Matrix Matrix::TaylorSeries(int order) {
	return this->Pow(order);
}

Matrix Matrix::DotProduct(Matrix element) {
	std::vector<std::vector<float>> mat = std::vector<std::vector<float>>();

	for (int i = 0; i < ColumnCount; i++) {
		mat.push_back(element.Multiply(this->Column(i)).RowSums());
	}

	return mat;
}

std::vector<float> Matrix::LogSumExp() {
	std::vector<float> logSum = std::vector<float>(ColumnCount);

	for (int c = 0; c < ColumnCount; c++) {

		std::vector<float> col = Column(c);

		auto maxElement = std::max_element(col.begin(), col.end());
		float max = *maxElement;
		float sum = 0;

		for (int i = 0; i < col.size(); i++) {
			sum += std::exp(col[i] - max);
		}
		logSum[c] = max + std::log(sum);
	}
	return logSum;
}

// Basic Math
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

Matrix Matrix::Sqrt() {
	return SingleFloatOperation(0, &sqrt);
}

Matrix Matrix::Sin() {
	return SingleFloatOperation(0, &sin);
}
Matrix Matrix::Cos() {
	return SingleFloatOperation(0, &cos);
}

// Activation functions and derivatives
Matrix Matrix::ReLU() {
	return SingleFloatOperation(0, &relu);
}
Matrix Matrix::ReLUDeriv() {
	return SingleFloatOperation(0, relu_derivative);
}

Matrix Matrix::LeakyReLU(float alpha) {
	return SingleFloatOperation(alpha, &leakyrelu);
}
Matrix Matrix::LeakyReLUDeriv(float alpha) {
	return SingleFloatOperation(alpha, &leakyrelu_derivative);
}


// Math implementations
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

Matrix Matrix::VectorFloatOperation(std::vector<float> scalar, void(*operation)(float* matrix, float* element, int num_elements)) {

	Matrix mat;
	Matrix element = Matrix(RowCount, scalar);

	if (scalar.size() == RowCount) {
		mat = this->Transpose();
	} else if (scalar.size() == ColumnCount) {
		mat = matrix;
	}

	std::vector<float> flatMat = mat.FlattenMatrix();
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

	return mat.ReshapeMatrix(flatMat);
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


// Shape
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