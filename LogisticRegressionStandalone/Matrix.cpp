#include "Matrix.h"

// Constructors

Matrix::Matrix() {
	matrix = std::vector<std::vector<float>>(0);
	ColumnCount = 0;
	RowCount = 0;
	transposeBuilt = false;
}
Matrix::Matrix(int rows, int columns) {
	matrix.reserve(rows);

	for (int i = 0; i < rows; i++) {
		matrix.emplace_back(columns);
	}

	ColumnCount = columns;
	RowCount = rows;
	transposeBuilt = false;
}
Matrix::Matrix(int rows, int columns, float value) {
	matrix.reserve(rows);

	for (int i = 0; i < rows; i++) {
		matrix.emplace_back(columns, value);
	}

	ColumnCount = columns;
	RowCount = rows;
	transposeBuilt = false;
}
Matrix::Matrix(int rows, int columns, init initType) {
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
	} else if (initType == Matrix::init::He) {
		std::normal_distribution<float> dist(0.0, std::sqrt(2.0f / RowCount));

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen);
			}
		}
	} else if (initType == Matrix::init::Normalize) {
		std::uniform_real_distribution<float> dist(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen);
			}
		}
	} else if (initType == Matrix::init::Random) {
		std::uniform_real_distribution<float> dist(lowerRand, upperRand);

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrix[r][c] = dist(gen) * std::sqrt(1.0f / ColumnCount);
			}
		}
	}

	transposeBuilt = false;
}
Matrix::Matrix(std::vector<std::vector<float>> matrix) {
	this->matrix = matrix;
	ColumnCount = matrix[0].size();
	RowCount = matrix.size();
	transposeBuilt = false;
}

// Util

void Matrix::SetColumn(int index, std::vector<float> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
}
void Matrix::SetColumn(int index, std::vector<int> vector) {
	for (int i = 0; i < RowCount; i++) {
		matrix[i][index] = vector[i];
	}

	transposeBuilt = false;
}

void Matrix::SetRow(int index, std::vector<float> vector) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
}
void Matrix::SetRow(int index, std::vector<int> vector) {
	for (int i = 0; i < ColumnCount; i++) {
		matrix[index][i] = vector[i];
	}

	transposeBuilt = false;
}

void Matrix::Insert(int startRow, Matrix element) {
	for (int i = 0; i < element.RowCount; i++) {
		this->SetRow(i + startRow, element.Row(i));
	}

	transposeBuilt = false;
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

std::vector<float> Matrix::ColumnSums() {
	std::vector<float> sums = std::vector<float>();
	sums.reserve(ColumnCount);

	if (transposeBuilt) {
		for (int r = 0; r < matrixT.size(); r++) {
			sums.push_back(std::reduce(matrixT[r].begin(), matrixT[r].end()));
		}
	}
	else {
		for (int c = 0; c < ColumnCount; c++) {
			sums.push_back(0);
			for (int r = 0; r < RowCount; r++) {
				sums[c] += matrix[r][c];
			}
		}
	}
	return sums;
}
std::vector<float> Matrix::RowSums() {
	std::vector<float> sums = std::vector<float>();
	sums.reserve(matrix.size());

	for (int r = 0; r < matrix.size(); r++) {
		sums.push_back(std::reduce(matrix[r].begin(), matrix[r].end()));
	}

	return sums;
}

std::vector<float> Matrix::Column(int index) {

	if (transposeBuilt) {
		return matrixT[index];
	} else {
		std::vector<float> column = std::vector<float>();
		column.reserve(RowCount);

		for (int i = 0; i < RowCount; i++) {
			column.push_back(matrix[i][index]);
		}
		return column;
	}
}
std::vector<float> Matrix::Row(int index) {
	return matrix[index];
}

// "Advanced" math
Matrix Matrix::ExtractFeatures(int fourier, int taylor, int chebyshev, int legendre, int laguerre, float lowerNormal, float upperNormal) {
	// Normalize
	Matrix mat = this->matrix;
	Matrix taylorNormal;
	Matrix fourierNormal;
	Matrix chebyshevNormal;

	if (fourier) { fourierNormal = mat.Normalized(-M_PI, M_PI); }
	if (taylor) { taylorNormal = mat.Normalized(0.0f, 1.0f); }
	if (chebyshev + legendre + laguerre) { chebyshevNormal = mat.Normalized(-1.0f, 1.0f); }

	// Compute Fourier Series
	if (fourier) {
		for (int f = 0; f < fourier; f++) {
			mat = mat.Combine(fourierNormal.FourierSeries(f + 1));
		}
	}

	// Compute Taylor Series
	if (taylor) {
		for (int t = 0; t < taylor; t++) {
			mat = mat.Combine(taylorNormal.TaylorSeries(t + 1));
		}
	}

	// Compute Chebyshev Series
	if (chebyshev) {
		for (int c = 0; c < chebyshev; c++) {
			mat = mat.Combine(chebyshevNormal.ChebyshevSeries(c + 1));
		}
	}

	// Compute Legendre Series
	if (legendre) {
		for (int l = 0; l < legendre; l++) {
			mat = mat.Combine(chebyshevNormal.LegendreSeries(l + 1));
		}
	}

	// Compute Laguerre Series
	if (laguerre) {
		for (int l = 0; l < laguerre; l++) {
			mat = mat.Combine(chebyshevNormal.LaguerreSeries(l + 1));
		}
	}

	mat = mat.Normalized(lowerNormal, upperNormal);

	return mat;
}

Matrix Matrix::Normalized(float lowerRange, float upperRange) {

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
	return this->Multiply(order).Sin().Combine(this->Multiply(order).Cos());
}
Matrix Matrix::TaylorSeries(int order) {
	return this->Pow(order);
}
Matrix Matrix::ChebyshevSeries(int order) {
	return this->Acos().Multiply(order).Cos();
}
Matrix Matrix::LegendreSeries(int order) {
	return (this->Pow(2) - 1).Pow(order);
}
Matrix Matrix::LaguerreSeries(int order) {
	return this->Pow(order).Multiply(this->Negative().Exp());
}

Matrix Matrix::DotProduct(Matrix element) {

	Matrix mat = Matrix(element.RowCount, ColumnCount);

	for (int i = 0; i < ColumnCount; i++) {
		mat.SetColumn(i, element.Multiply(this->Column(i)).RowSums());
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
Matrix Matrix::Negative() {
	return SingleFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, -1);
}
Matrix Matrix::Abs() {
	return SingleFloatOperation(&Matrix::SIMDAbs, &Matrix::RemainderAbs, 0);
}

Matrix Matrix::Add(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, scalar);
}
Matrix Matrix::Add(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDAdd, &Matrix::RemainderAdd, element);
}

Matrix Matrix::Subtract(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, scalar);
}
Matrix Matrix::Subtract(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDSub, &Matrix::RemainderSub, element);
}

Matrix Matrix::Multiply(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, scalar);
}
Matrix Matrix::Multiply(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDMul, &Matrix::RemainderMul, element);
}

Matrix Matrix::Divide(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, scalar);
}
Matrix Matrix::Divide(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDDiv, &Matrix::RemainderDiv, element);
}

Matrix Matrix::Pow(float scalar) {
	return SingleFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
Matrix Matrix::Pow(std::vector<float> scalar) {
	return VectorFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, scalar);
}
Matrix Matrix::Pow(Matrix element) {
	return MatrixFloatOperation(&Matrix::SIMDPow, &Matrix::RemainderPow, element);
}

Matrix Matrix::Exp(float base) {
	return SingleFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
Matrix Matrix::Exp(std::vector<float> base) {
	return VectorFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}
Matrix Matrix::Exp(Matrix base) {
	return MatrixFloatOperation(&Matrix::SIMDExp, &Matrix::RemainderExp, base);
}

Matrix Matrix::Log(float base) {
	return matrix;
}

Matrix Matrix::Cos() {
	return this->SingleFloatOperation(&Matrix::SIMDCos, &Matrix::RemainderCos, 0);
}
Matrix Matrix::Sin() {
	return this->SingleFloatOperation(&Matrix::SIMDSin, &Matrix::RemainderSin, 0);
}

Matrix Matrix::Acos() {
	return this->SingleFloatOperation(&Matrix::SIMDAcos, &Matrix::RemainderAcos, 0);
}
Matrix Matrix::Asin() {
	return this->SingleFloatOperation(&Matrix::SIMDAsin, &Matrix::RemainderAcos, 0);
}

// Activation Functions
Matrix Matrix::Sigmoid() {
	Matrix one = Matrix(RowCount, ColumnCount, 1);
	return one / (this->Negative().Exp() + 1);
}
Matrix Matrix::ReLU() {
	return SingleFloatOperation(&Matrix::SIMDMax, &Matrix::RemainderMax, 0);
}
Matrix Matrix::LeakyReLU(float alpha) {
	return MatrixFloatOperation(&Matrix::SIMDMax, &Matrix::RemainderMax, this->Multiply(alpha));
}
Matrix Matrix::_LeakyReLU() {
	return LeakyReLU();
}
Matrix Matrix::ELU(float alpha) {
	Matrix a = matrix;
	for (int r = 0; r < this->RowCount; r++) {
		for (int c = 0; c < this->ColumnCount; c++) {
			a[r][c] = matrix[r][c] < 0.0f ? alpha * (std::exp(matrix[r][c]) - 1) : matrix[r][c];
		}
	}
	return a;
}
Matrix Matrix::_ELU() {
	return ELU();
}
Matrix Matrix::Tanh() {
	Matrix a = this->Exp();
	Matrix b = this->Negative().Exp();

	return (a - b) / (a + b);
}
Matrix Matrix::Softplus() {
	return (this->Exp() + 1).Log();
}
Matrix Matrix::SiLU() {
	return this->Divide((this->Negative().Exp() + 1));
}

Matrix Matrix::SoftMax() {
	return this->Subtract(this->LogSumExp()).Exp();
}

// Activation Derivatives
Matrix Matrix::SigmoidDerivative() {
	Matrix a = matrix;
	return this->Sigmoid() * (a - this->Sigmoid());
}
Matrix Matrix::ReLUDerivative() {
	Matrix a = matrix;

	for (int r = 0; r < this->RowCount; r++) {
		for (int c = 0; c < this->ColumnCount; c++) {
			a[r][c] = matrix[r][c] > 0.0f ? 1.0f : 0.0f;
		}
	}
	return a;
}
Matrix Matrix::LeakyReLUDerivative(float alpha) {
	Matrix deriv = matrix;
	for (int c = 0; c < ColumnCount; c++) {
		for (int r = 0; r < RowCount; r++) {
			deriv[r][c] = deriv[r][c] > 0.0f ? 1.0f : alpha;
		}
	}
	return deriv;
}
Matrix Matrix::_LeakyReLUDerivative() {
	return LeakyReLUDerivative();
}
Matrix Matrix::ELUDerivative(float alpha) {
	Matrix a = matrix;

	for (int r = 0; r < this->RowCount; r++) {	
		for (int c = 0; c < this->ColumnCount; c++) {
			a[r][c] = matrix[r][c] > 0.0f ? 1.0f : alpha * std::exp(matrix[r][c]);
		}
	}
	return a;
}
Matrix Matrix::_ELUDerivative() {
	return ELUDerivative();
}
Matrix Matrix::TanhDerivative() {
	Matrix one = Matrix(this->RowCount, this->ColumnCount, 1);
	return one - this->Tanh().Pow(2);
}
Matrix Matrix::SoftplusDerivative() {
	Matrix one = Matrix(this->RowCount, this->ColumnCount, 1);
	return one / (this->Negative().Exp() + 1);
}
Matrix Matrix::SiLUDerivative() {
	return (this->Negative().Exp() + (this->Multiply(this->Negative().Exp()) + 1) / (this->Negative().Exp() + 1).Pow(2));
}

// SIMD Implementations
Matrix Matrix::SingleFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo),
	float (Matrix::* remainderOperation)(float a, float b), float scalar) {
	std::vector<std::vector<float>> mat = matrix;
	const int alignedN = mat[0].size() - (mat[0].size() % 8);
	__m256 _scalar = _mm256_set1_ps(scalar);

	for (int r = 0; r < mat.size(); r++) {

		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_load_ps(&mat[r][i]);
			loaded_a = (this->*operation)(loaded_a, _scalar);
			_mm256_store_ps(&mat[r][i], loaded_a);
		}

		for (int i = alignedN; i < mat[r].size(); i++) {
			mat[r][i] = (this->*remainderOperation)(mat[r][i], scalar);
		}
	}
	return mat;
}

Matrix Matrix::VectorFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo),
	float (Matrix::* remainderOperation)(float a, float b), std::vector<float> scalar) {

	Matrix mat = matrix;
	const int alignedN = mat.matrix[0].size() - (mat.matrix[0].size() % 8);

	if (scalar.size() == ColumnCount) {
		for (int r = 0; r < mat.matrix.size(); r++) {

			for (int i = 0; i < alignedN; i += 8) {
				__m256 loaded_a = _mm256_load_ps(&mat.matrix[r][i]);
				__m256 loaded_b = _mm256_load_ps(&scalar[i]);

				loaded_a = (this->*operation)(loaded_a, loaded_b);
				_mm256_store_ps(&mat.matrix[r][i], loaded_a);
			}

			for (int i = alignedN; i < mat.matrix[r].size(); i++) {
				mat.matrix[r][i] = (this->*remainderOperation)(mat.matrix[r][i], scalar[i]);
			}
		}
	} else if (scalar.size() == RowCount) {
		for (int r = 0; r < mat.matrix.size(); r++) {

			float value = scalar[r];

			__m256 loaded_b = _mm256_set1_ps(scalar[r]);

			for (int i = 0; i < alignedN; i += 8) {
				__m256 loaded_a = _mm256_load_ps(&mat.matrix[r][i]);

				loaded_a = (this->*operation)(loaded_a, loaded_b);
				_mm256_store_ps(&mat.matrix[r][i], loaded_a);
			}

			for (int i = alignedN; i < mat.matrix[r].size(); i++) {
				mat.matrix[r][i] = (this->*remainderOperation)(mat.matrix[r][i], value);
			}
		}
	}

	return mat;
}

Matrix Matrix::MatrixFloatOperation(__m256 (Matrix::* operation)(__m256 opOne, __m256 opTwo),
	float (Matrix::* remainderOperation)(float a, float b), Matrix element) {
	std::vector<std::vector<float>> mat = element.matrix;
	const int alignedN = mat[0].size() - (mat[0].size() % 8);

	for (int r = 0; r < mat.size(); r++) {
		for (int i = 0; i < alignedN; i += 8) {
			__m256 loaded_a = _mm256_load_ps(&matrix[r][i]);
			__m256 loaded_b = _mm256_load_ps(&mat[r][i]);

			loaded_a = (this->*operation)(loaded_a, loaded_b);
			_mm256_store_ps(&mat[r][i], loaded_a);
		}

		for (int i = alignedN; i < mat[r].size(); i++) {
			mat[r][i] = (this->*remainderOperation)(matrix[r][i], mat[r][i]);
		}
	}
	return mat;
}


// SIMD Operations

__m256 Matrix::SIMDAdd(__m256 opOne, __m256 opTwo) {
	return _mm256_add_ps(opOne, opTwo);
}
__m256 Matrix::SIMDSub(__m256 opOne, __m256 opTwo) {
	return _mm256_sub_ps(opOne, opTwo);
}
__m256 Matrix::SIMDMul(__m256 opOne, __m256 opTwo) {
	return _mm256_mul_ps(opOne, opTwo);
}
__m256 Matrix::SIMDDiv(__m256 opOne, __m256 opTwo) {
	return _mm256_div_ps(opOne, opTwo);
}
__m256 Matrix::SIMDPow(__m256 opOne, __m256 opTwo) {
	return _mm256_pow_ps(opOne, opTwo);
}
__m256 Matrix::SIMDExp(__m256 opOne, __m256 opTwo) {
	return _mm256_pow_ps(opTwo, opOne);
}
__m256 Matrix::SIMDMax(__m256 opOne, __m256 opTwo) {
	return _mm256_max_ps(opOne, opTwo);
}
__m256 Matrix::SIMDAbs(__m256 opOne, __m256 opTwo) {
	__m256 mask = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
	__m256 result = _mm256_and_ps(opOne, mask);
	return result;
}

// SIMD Trig
__m256 Matrix::SIMDSin(__m256 opOne, __m256 opTwo) {
	return _mm256_sin_ps(opOne);
}
__m256 Matrix::SIMDCos(__m256 opOne, __m256 opTwo) {
	return _mm256_cos_ps(opOne);
}
__m256 Matrix::SIMDSec(__m256 opOne, __m256 opTwo) {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_cos_ps(opOne));
}
__m256 Matrix::SIMDCsc(__m256 opOne, __m256 opTwo) {
	return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sin_ps(opOne));
}
__m256 Matrix::SIMDAcos(__m256 opOne, __m256 opTwo) {
	return _mm256_acos_ps(opOne);
}
__m256 Matrix::SIMDAsin(__m256 opOne, __m256 opTwo) {
	return _mm256_asin_ps(opOne);
}

float Matrix::RemainderAdd(float a, float b) {
	return a + b;
}
float Matrix::RemainderSub(float a, float b) {
	return a - b;
}
float Matrix::RemainderMul(float a, float b) {
	return a * b;
}
float Matrix::RemainderDiv(float a, float b) {
	return a / b;
}
float Matrix::RemainderPow(float a, float b) {
	return std::pow(a, b);
}
float Matrix::RemainderExp(float a, float b) {
	return std::pow(b, a);
}
float Matrix::RemainderMax(float a, float b) {
	return std::max(a, b);
}
float Matrix::RemainderAbs(float a, float b) {
	return std::abs(a);
}

// SIMD Trig
float Matrix::RemainderSin(float a, float b) {
	return std::sin(a);
}
float Matrix::RemainderCos(float a, float b) {
	return std::cos(a);
}
float Matrix::RemainderSec(float a, float b) {
	return 1.0f / std::cos(a);
}
float Matrix::RemainderCsc(float a, float b) {
	return 1.0f / std::sin(a);
}
float Matrix::RemainderAcos(float a, float b) {
	return std::acos(a);
}
float Matrix::RemainderAsin(float a, float b) {
	return std::asin(a);
}

// MISC

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
std::string Matrix::Size() {
	return std::to_string(RowCount).append(" :: ").append(std::to_string(ColumnCount)).append("\n");
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

Matrix Matrix::Join(Matrix element) {
	Matrix a = Matrix(RowCount, element.ColumnCount + ColumnCount);

	for (int r = 0; r < a.RowCount; r++) {
		for (int c = 0; c < a.ColumnCount; c++) {
			a[r][c] = (c < ColumnCount ? matrix[r][c] : element[r][c - ColumnCount]);
		}
	}
	return a;
}

Matrix Matrix::Transpose() {

	if (transposeBuilt) {
		return matrixT;
	}
	else {
		matrixT.reserve(ColumnCount);

		for (int i = 0; i < ColumnCount; i++) {
			matrixT.emplace_back(RowCount);
		}

		for (int r = 0; r < RowCount; r++) {
			for (int c = 0; c < ColumnCount; c++) {
				matrixT[c][r] = this->matrix[r][c];
			}
		}

		transposeBuilt = true;

		return matrixT;
	}
}