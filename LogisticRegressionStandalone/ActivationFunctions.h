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

Matrix SoftMax(Matrix total);