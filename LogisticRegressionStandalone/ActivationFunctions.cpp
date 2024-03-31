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


Matrix SoftMax(Matrix total) {
	return (total - total.LogSumExp()).Exp();
}
