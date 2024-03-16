#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>

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
Matrix SoftMax(Matrix total);
Matrix ReLUDerivative(Matrix total);
void InitializeNetwork();
void InitializeResultMatrices(int size);
void TrainNetwork();
void ForwardPropogation();
void BackwardPropogation();
void UpdateNetwork();
void LoadInput();
int ReadBigInt(ifstream* fr);
Matrix RandomizeInput(Matrix totalInput, int size);
vector<int> GetPredictions(int len);
float Accuracy(vector<int> predictions, vector<int> labels);

int main()
{
	LoadInput();
	
	InitializeNetwork();
	
	TrainNetwork();

	return 0;
}

void LoadInput() {

	auto sTime = std::chrono::high_resolution_clock::now();

	string trainingImages = "Training Data\\train-images.idx3-ubyte";
	string trainingLabels = "Training Data\\train-labels.idx1-ubyte";

	ifstream trainingFR = ifstream(trainingImages, std::ios::binary);
	ifstream trainingLabelsFR = ifstream(trainingLabels, std::ios::binary);

	if (trainingFR.is_open() && trainingLabelsFR.is_open()) {
		cout << "Loading training data..." << endl;
	}
	else {
		std::cout << "File(s) not found" << endl;
	}

	// Discard
	int magicNum = ReadBigInt(&trainingLabelsFR);
	int imageNum = ReadBigInt(&trainingLabelsFR);
	magicNum = ReadBigInt(&trainingFR);

	// Read the important things
	imageNum = ReadBigInt(&trainingFR);
	int width = ReadBigInt(&trainingFR);
	int height = ReadBigInt(&trainingFR);

	input = Matrix((width * height), imageNum);
	inputLabels = vector<int>(imageNum);
	YTotal = Matrix(outputLayerSize, imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		vector<int> y = vector<int>(outputLayerSize, 0);

		char byte;
		trainingLabelsFR.read(&byte, 1);
		int label = static_cast<int>(static_cast<unsigned char>(byte));

		inputLabels[i] = label;

		y[label] = 1;
		YTotal.SetColumn(i, y);
	}

	input = input.Divide(255);

	trainingFR.close();
	trainingLabelsFR.close();

	auto eTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time = eTime - sTime;

	cout << "Time to load input " << (time.count() / 1000.00) << " seconds" << endl;
}

int ReadBigInt(ifstream* fr) {

	int littleInt;
	fr->read(reinterpret_cast<char*>(&littleInt), sizeof(int));

	unsigned char* bytes = reinterpret_cast<unsigned char*>(&littleInt);
	std::swap(bytes[0], bytes[3]);
	std::swap(bytes[1], bytes[2]);

	return littleInt;
}

void InitializeNetwork() {

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
	dTotal = vector<Matrix>(aTotal.size());

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = Matrix(weights[i].ColumnCount, size);
		activation[i] = Matrix(aTotal[i].RowCount, size);
		dTotal[i] = Matrix(aTotal[i].RowCount, size);
	}
}

Matrix RandomizeInput(Matrix totalInput, int size) {
	Matrix a = Matrix(totalInput.RowCount, size);

	std::unordered_set<int> used = std::unordered_set<int>(size);

	YBatch = Matrix(outputLayerSize, size);
	batchLabels.clear();

	srand(time(0));

	while (batchLabels.size() < size) {

		int c = (rand() % totalInput.ColumnCount) + 1;

		if (used.find(c) == used.end()) {

			used.insert(c);

			a.SetColumn(batchLabels.size(), totalInput.Column(c));
			YBatch.SetColumn(batchLabels.size(), YTotal.Column(c));
			batchLabels.push_back(inputLabels[c]);
		}
	}

	return a;
}

void TrainNetwork() {

	cout << "TRAINING STARTED" << endl;

	std::chrono::steady_clock::time_point tStart;
	std::chrono::steady_clock::time_point tEnd;
	std::chrono::duration<double, std::milli> time;

	batch = RandomizeInput(input, batchSize);

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

void ForwardPropogation() {

	/*cout << "batch: " << batch.RowCount << " :: " << batch.ColumnCount << endl;
	for (int i = 0; i < activation.size(); i++) {
		cout << "A [" << i << "] " << activation[i].RowCount << " :: " << activation[i].ColumnCount << endl;
		cout << "ATotal [" << i << "] " << aTotal[i].RowCount << " :: " << aTotal[i].ColumnCount << endl;
		cout << "Weights [" << i << "] " << weights[i].RowCount << " :: " << weights[i].ColumnCount << endl;
	}
	cout << endl;*/

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].DotProduct(i == 0 ? batch : activation[i - 1]) + biases[i]);
		aTotal[i] = aTotal[i].Transpose();
		activation[i] = i < aTotal.size() - 1 ? ReLU(aTotal[i]) : SoftMax(aTotal[i]);
	}

	/*cout << "batch: " << batch.RowCount << " :: " << batch.ColumnCount << endl;
	for (int i = 0; i < activation.size(); i++) {
		cout << "A [" << i << "] " << activation[i].RowCount << " :: " << activation[i].ColumnCount << endl;
		cout << "ATotal [" << i << "] " << aTotal[i].RowCount << " :: " << aTotal[i].ColumnCount << endl;
	}
	cout << endl;*/
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] -= YBatch;
	cout << "DHot complete" << endl;

	for (int i = dTotal.size() - 2; i > -1; i--) {
		cout << "DTotal [" << i << "] " << dTotal[i].RowCount << " :: " << activation[i].ColumnCount << endl;
		cout << "DTotal [" << i + 1 << "] " << dTotal[i + 1].RowCount << " :: " << activation[i].ColumnCount << endl;
		cout << "ATotal [" << i << "] " << aTotal[i].RowCount << " :: " << aTotal[i].ColumnCount << endl;
		cout << "ReluDeriv: " << ReLUDerivative(aTotal[i]).RowCount << " :: " << ReLUDerivative(aTotal[i]).ColumnCount << endl;
		cout << "Weights [" << i + 1 << "] " << weights[i + 1].RowCount << " :: " << weights[i].ColumnCount << endl;

		dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])) * ReLUDerivative(aTotal[i]));
	}
	cout << "DTotal Complete" << endl;

	/*
	Expected:

	dW1 = 784 x 128
	dT1 = 128 x 500
	input = 784 x 500
	
	dW2 128 x 10
	dT2 = 10 x 500
	a1 = 128 x 500
	*/

	for (int i = 0; i < weights.size(); i++) {
		cout << i << endl;
		cout << "dW: " << dWeights[i].RowCount << " :: " << dWeights[i].ColumnCount << endl;
		cout << "dT: " << dTotal[i].RowCount << " :: " << dTotal[i].ColumnCount << endl;
		if (i == 1) { cout << "A: " << activation[i - 1].RowCount << " :: " << activation[i - 1].ColumnCount << endl; }
		else { cout << "batch: " << batch.RowCount << " :: " << batch.ColumnCount << endl; }

		dWeights[i] = dTotal[i].DotProduct(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize);
	}
	cout << "DWeights Complete" << endl;

	for (int i = 0; i < biases.size(); i++) {
		dBiases[i] = dTotal[i].MultiplyAndSum(1.0f / (float)batchSize);
	}
	cout << "DBias Complete" << endl;
}

void UpdateNetwork() {
	for (int i = 0; i < weights.size(); i++) {
		weights[i] -= dWeights[i] * learningRate;
	}

	for (int i = 0; i < biases.size(); i++) {
		for (int x = 0; x < biases[i].size(); x++) {
			biases[i][x] -= dBiases[i][x] * learningRate;
		}
	}
}

vector<int> GetPredictions(int len) {

	vector<int> predictions = vector<int>(len);

	for (int i = 0; i < len; i++) {

		vector<float> a = activation[activation.size() - 1].Column(i);

		auto maxElementIterator = std::max_element(a.begin(),
			a.end());
		predictions[i] = std::distance(a.begin(), maxElementIterator);
	}
	return predictions;
}

float Accuracy(vector<int> predictions, vector<int> labels) {
	int correct = 0;

	for (int i = 0; i < predictions.size(); i++)
	{
		if (predictions[i] == labels[i])
		{
			correct++;
		}
	}
	return (float)correct / (float)predictions.size();
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

Matrix SoftMax(Matrix total) {

	Matrix softmax = total;
	softmax = total.Exp() / total.Exp().ColumnSums();

	return softmax;
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