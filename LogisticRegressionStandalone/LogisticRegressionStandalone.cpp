#include <iostream>

#include "Matrix.h"

using namespace std;

// Inputs
Matrix input;
Matrix batch;

// Neural Network Matrices
vector<Matrix> weights;
Matrix biases;

vector<Matrix> activation;
vector<Matrix> aTotal;

vector<Matrix> dTotal;
vector<Matrix> dWeights;
Matrix dBiases;

// Prototype

Matrix ReLU(Matrix total);

int main()
{

}

void InitializeNetwork() {

}

void TrainNetwork() {

}

void ForwardPropogation() {
	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i] * (i == 0 ? batch : activation[i - 1])) + biases[i];
		activation[i] = ReLU(aTotal[i]);
	}
}

void BackwardPropogation() {

}

void UpdateNetwork() {

}

Matrix ReLU(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] < 0.0f ? 0.0f : total[r][c];
		}
	}
	return a;
}