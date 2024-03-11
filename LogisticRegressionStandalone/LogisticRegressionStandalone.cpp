#include <iostream>

#include "Matrix.h"

using namespace std;

// Hyperparameters
int inputLayerSize = 784;
int outputLayerSize = 10;
vector<int> hiddenSize = {128};

float learningRate;
float thresholdAccuracy;
int batchSize;

// Inputs
Matrix input;
Matrix batch;

// Neural Network Matrices
vector<Matrix> weights;
vector<vector<float>> biases;

vector<Matrix> activation;
vector<Matrix> aTotal;

vector<Matrix> dTotal;
vector<Matrix> dWeights;
Matrix dBiases;

// Prototypes

Matrix ReLU(Matrix total);
void InitializeNetwork();
void InitializeResultMatrices(int size);

int main()
{
	InitializeNetwork();
}

void InitializeNetwork() {
	int connections = 0;

	weights = vector<Matrix>(hiddenSize.size() + 1);
	dWeights = vector<Matrix>(weights.size());

	biases = vector<vector<float>>(weights.size());
	dBiases = vector<vector<float>>(biases.size());

	for (int i = 0; i < weights.size(); i++) {
		weights[i] = Matrix(i == 0 ? inputLayerSize : hiddenSize[i - 1], i == weights.size() - 1 ? outputLayerSize : hiddenSize[i], -0.5f, 0.5f);
		cout << "Weights[" << i << "] connections: " << (weights[i].ColumnCount * weights[i].RowCount) << endl;
		connections += weights[i].ColumnCount * weights[i].RowCount;

		biases[i] = vector<float>(weights[i].Row(i));
		cout << "Biases[" << i << "] connections: " << biases[i].size() << endl;
		connections += biases[i].size();

		dWeights[i] = Matrix(weights[i].RowCount, weights[i].ColumnCount);
		dBiases[i] = vector<float>(biases[i].size());
	}

	cout << "Total connections: " << connections << endl;

	InitializeResultMatrices(batchSize);

	cout << "INITIALIZATION COMPLETE" << endl;
}

void InitializeResultMatrices(int size) {
	aTotal = vector<Matrix>(weights.size());
	activation = vector<Matrix>(aTotal.size());

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = Matrix(weights[i].ColumnCount, size);
		activation[i] = Matrix(aTotal[i].RowCount, aTotal[i].ColumnCount);
	}
}

void TrainNetwork() {

}

void ForwardPropogation() {
	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].CollapseAndLeftMultiply(i == 0 ? batch : activation[i - 1])) + biases[i];
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