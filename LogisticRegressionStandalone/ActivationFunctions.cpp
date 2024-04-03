#include "ActivationFunctions.h"

Matrix ReLU(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] < 0.0f ? 0.0f : total[r][c];
		}
	}
	return a;
}

Matrix ReLUDerivative(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] > 0.0f ? 1.0f : 0.0f;
		}
	}
	return a;
}


Matrix LeakyReLU(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] < 0.0f ? (0.1f * total[r][c]) : total[r][c];
		}
	}
	return a;
}

Matrix LeakyReLUDerivative(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] > 0.0f ? 1.0f : 0.1f;
		}
	}
	return a;
}


Matrix ELU(Matrix total, float alpha) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] < 0.0f ? alpha * (std::exp(total[r][c] - 1)) : total[r][c];
		}
	}
	return a;
}

Matrix ELUDerivative(Matrix total, float alpha) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] > 0.0f ? 1.0f : alpha * std::exp(total[r][c]);
		}
	}
	return a;
}


Matrix Tanh(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = (std::exp(total[r][c]) - std::exp(-total[r][c])) / (std::exp(total[r][c]) + std::exp(-total[r][c]));
		}
	}
	return a;
}

Matrix TanhDerivative(Matrix total) {
	Matrix a = Matrix(total.RowCount, total.ColumnCount, 1);
	return a - Tanh(total).Pow(2);
}


Matrix Sigmoid(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = 1 / (1 + std::exp(-total[r][c]));
		}
	}
	return a;
}

Matrix SigmoidDerivative(Matrix total) {
	Matrix a = Matrix(total.RowCount, total.ColumnCount, 1);
	return Sigmoid(total) * (a - Sigmoid(total));
}


//Matrix Smht(Matrix total) {
//
//}
//
//Matrix SmhtDerivative(Matrix total) {
//
//}
//
//
//Matrix GELU(Matrix total) {
//	Matrix a = total;
//	for (int r = 0; r < total.RowCount; r++) {
//		for (int c = 0; c < total.ColumnCount; c++) {
//			a[r][c] = (0.5 * total[r][c]) * (1 + std::erf(total[r][c] / std::sqrt(2)));
//		}
//	}
//	return a;
//}
//
//Matrix GELUDerivative(Matrix total) {
//
//}


Matrix Softplus(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = std::log(1 + std::exp(total[r][c]));
		}
	}
	return a;
}

Matrix SoftplusDerivative(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = 1 / (1 + std::exp(-total[r][c]));
		}
	}
	return a;
}


Matrix SiLU(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] / (1 + std::exp(-total[r][c]));
		}
	}
	return a;
}

Matrix SiLUDerivative(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = (1 + std::exp(-total[r][c]) + (total[r][c] * std::exp(-total[r][c]))) / std::pow(1 + std::exp(-total[r][c]), 2);
		}
	}
	return a;
}


Matrix Gaussian(Matrix total) {
	return total.Pow(2).Multiply(-1).Exp();
}

Matrix GaussianDerivative(Matrix total) {
	Matrix a = total;
	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = (-2) * total[r][c] * (std::exp(-std::pow(total[r][c], 2)));
		}
	}
	return a;
}


Matrix Swish(Matrix total) {
	return total * Sigmoid(total);
}

Matrix SwishDerivative(Matrix total) {
	Matrix a = Matrix(total.RowCount, total.ColumnCount, 1);
	return Sigmoid(total) + total * Sigmoid(total) * (a - Sigmoid(total));
}


Matrix Cubic(Matrix total) {
	return total.Multiply(1/3).Pow(3);
}

Matrix CubicDerivative(Matrix total) {
	return total.Pow(2).Divide(9);
}


Matrix SoftMax(Matrix total) {
	return (total - total.LogSumExp()).Exp();
}
