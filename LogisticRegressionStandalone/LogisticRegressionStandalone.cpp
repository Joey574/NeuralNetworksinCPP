#include <iostream>
#include <chrono>

#include "Matrix.h"

using namespace std;

// Hyperparameters
int inputLayerSize = 784;
int outputLayerSize = 10;
vector<int> hiddenSize = {128};

float learningRate = 0.1f;
float thresholdAccuracy = 0.25f;
int batchSize = 500;
int iterations = 500;

// Inputs
Matrix input;
Matrix batch;

vector<int> inputLabels;
vector<int> batchLabels;

// Neural Network Matrices
vector<Matrix> weights;
vector<vector<float>> biases;

vector<Matrix> activation;
vector<Matrix> aTotal;

vector<Matrix> dTotal;
vector<Matrix> dWeights;
vector<vector<float>> dBiases;

// Error stuff
Matrix YTotal;
Matrix YBatch;

// Prototypes

Matrix ReLU(Matrix total);
void InitializeNetwork();
void InitializeResultMatrices(int size);
void ForwardPropogation();
void BackwardPropogation();
void UpdateNetwork();
Matrix RandomizeInput(Matrix totalInput, int size);
vector<float> GetPredictions(int len);
float Accuracy(vector<float> predictions, vector<int> labels);

int main()
{
	InitializeNetwork();

	return 0;
}

void InitializeNetwork()   {

	auto initStart = std::chrono::high_resolution_clock::now();

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

	double floatSize = (sizeof(float)) * connections;
	double commas = connections - 1;
	double newLines = 0;

	for (int i = 0; i < weights.size(); i++) {
		newLines += weights[i].RowCount;
	}

	double fileSize = floatSize + commas + newLines;

	cout << "Predicted size of file: " << (fileSize / 1000000) << "mb" << endl;

	InitializeResultMatrices(batchSize);

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
	cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << endl;
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

	std::chrono::steady_clock::time_point tStart;
	std::chrono::steady_clock::time_point tEnd;
	std::chrono::duration<double, std::milli> time;

	for (int i = 0; i < iterations; i++) {
		float acc = Accuracy(GetPredictions(batchSize), batchLabels);

		if (acc > thresholdAccuracy) {
			batch = RandomizeInput(input, batchSize);
		}

		cout << "Iteration: " << i << " Accuracy: " << acc << endl;

		tStart = std::chrono::high_resolution_clock::now();
		ForwardPropogation();
		tEnd = std::chrono::high_resolution_clock::now();
		time = tEnd - tStart;
		cout << "Forward Propogation complete (" << time.count() << "ms)" << endl;

		tStart = std::chrono::high_resolution_clock::now();
		BackwardPropogation();
		tEnd = std::chrono::high_resolution_clock::now();
		time = tEnd - tStart;
		cout << "Backward Propogation complete (" << time.count() << "ms)" << endl;

		UpdateNetwork();
	}
}

Matrix RandomizeInput(Matrix totalInput, int size) {
	return totalInput;
}

void ForwardPropogation() {
	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].CollapseAndLeftMultiply(i == 0 ? batch : activation[i - 1])) + biases[i];
		activation[i] = ReLU(aTotal[i]);
	}
}

void BackwardPropogation() {
	dTotal[dTotal.size() - 1] = dTotal[dTotal.size() - 1] - YBatch;

	for (int i = dTotal.size() - 2; i > -1; i--) {

	}

	for (int i = 0; i < weights.size(); i++) {

	}

	for (int i = 0; i < biases.size(); i++) {
		dBiases[i] = dTotal[i].MultiplyAndSum(1.0f / (float)batchSize);
	}
}

vector<float> GetPredictions(int len) {
	vector<float> predictions = vector<float>(len);
	for (int i = 0; i < len; i++) {
		auto maxElementIterator = std::max_element(activation[activation.size() - 1].Column(i).begin(), 
			activation[activation.size() - 1].Column(i).end());
		predictions[i] = std::distance(activation[activation.size() - 1].Column(i).begin(), maxElementIterator);
	}
	return predictions;
}

float Accuracy(vector<float> predictions, vector<int> labels) {
	int correct = 0;

	for (int i = 0; i < predictions.size(); i++)
	{
		if ((int)(predictions[i] + 0.1f) == labels[i])
		{
			correct++;
		}
	}
	return (float)correct / predictions.size();
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