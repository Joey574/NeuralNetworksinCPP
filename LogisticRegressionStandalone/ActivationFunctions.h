#pragma once

#include "Matrix.h"

Matrix ReLU(Matrix total);
Matrix ReLUDerivative(Matrix total);

Matrix LeakyReLU(Matrix total);
Matrix LeakyReLUDerivative(Matrix total);

Matrix ELU(Matrix total, float alpha = 1.0f);
Matrix ELUDerivative(Matrix total, float alpha = 1.0f);

Matrix Tanh(Matrix total);
Matrix TanhDerivative(Matrix total);

Matrix Sigmoid(Matrix total);
Matrix SigmoidDerivative(Matrix total);

Matrix Smht(Matrix total);
Matrix SmhtDerivative(Matrix total);

Matrix GELU(Matrix total);
Matrix GELUDerivative(Matrix total);

Matrix Softplus(Matrix total);
Matrix SoftplusDerivative(Matrix total);

Matrix SiLU(Matrix total);
Matrix SiLUDerivative(Matrix total);

Matrix Gaussian(Matrix total);
Matrix GaussianDerivative(Matrix total);

Matrix SoftMax(Matrix total);