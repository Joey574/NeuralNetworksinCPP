#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>

#include "Matrix.h"

using namespace std;

// Hyperparameters
int inputLayerSize = 784;
int outputLayerSize = 10;
vector<int> hiddenSize = { 128 };

float learningRate = 0.05f;
float thresholdAccuracy = 0.15f;
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
void TestNetwork();
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
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	srand(time(0));

	LoadInput();

	InitializeNetwork();

	TrainNetwork();

	TestNetwork();

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

	trainingFR.close();
	trainingLabelsFR.close();

	input = input.Divide(255);

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

	std::chrono::steady_clock::time_point totalStart;
	std::chrono::steady_clock::time_point totalEnd;

	std::chrono::steady_clock::time_point tStart;
	std::chrono::steady_clock::time_point tEnd;
	std::chrono::duration<double, std::milli> time;

	totalStart = std::chrono::high_resolution_clock::now();

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

	totalEnd = std::chrono::high_resolution_clock::now();

	time = (totalEnd - totalStart);

	float avgTime = time.count() / iterations;

	time /= 1000.00;

	cout << "Total Training Time: " << time.count() << " seconds :: " << (time.count() / 60.00) << " minutes :: " << (time.count() / 3600.00) << " hours" << endl;
	cout << "Average Iteration Time: " << avgTime << " ms" << endl;
}

void ForwardPropogation() {

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].DotProduct(i == 0 ? batch : activation[i - 1]) + biases[i]).Transpose();
		activation[i] = i < aTotal.size() - 1 ? ReLU(aTotal[i]) : SoftMax(aTotal[i]);
	}
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - YBatch;

	for (int i = dTotal.size() - 2; i > -1; i--) {
		dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])).Transpose() * ReLUDerivative(aTotal[i]));
	}

	for (int i = 0; i < weights.size(); i++) {
		dWeights[i] = (dTotal[i].Transpose().DotProduct(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize)).Transpose();
		dBiases[i] = dTotal[i].MultiplyAndSum(1.0f / (float)batchSize);
	}
}

void UpdateNetwork() {
	for (int i = 0; i < weights.size(); i++) {
		weights[i] -= dWeights[i].Multiply(learningRate);
	}

	for (int i = 0; i < biases.size(); i++) {
		for (int x = 0; x < biases[i].size(); x++) {
			biases[i][x] -= (dBiases[i][x] * learningRate);
		}
	}
}

void TestNetwork() {
	string testingImages = "Testing Data\\t10k-images.idx3-ubyte";
	string testingLabels = "Testing Data\\t10k-labels.idx1-ubyte";

	ifstream testingFR = ifstream(testingImages, std::ios::binary);
	ifstream testingLabelFR = ifstream(testingLabels, std::ios::binary);

	if (testingFR.is_open() && testingLabelFR.is_open()) {
		cout << "Loading testing data..." << endl;
	}
	else {
		std::cout << "File(s) not found" << endl;
	}

	// Discard
	int magicNum = ReadBigInt(&testingLabelFR);
	int imageNum = ReadBigInt(&testingLabelFR);
	magicNum = ReadBigInt(&testingFR);

	// Read the important things
	imageNum = ReadBigInt(&testingFR);
	int width = ReadBigInt(&testingFR);
	int height = ReadBigInt(&testingFR);

	input = Matrix((width * height), imageNum);
	inputLabels = vector<int>(imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		testingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		char byte;
		testingLabelFR.read(&byte, 1);
		inputLabels[i] = static_cast<int>(static_cast<unsigned char>(byte));
	}
	testingFR.close();
	testingLabelFR.close();

	input = input.Divide(255);

	InitializeResultMatrices(input.ColumnCount);

	batch = input;

	ForwardPropogation();

	float acc = Accuracy(GetPredictions(input.ColumnCount), inputLabels);

	cout << "Final Accuracy: " << acc << endl;
}

vector<int> GetPredictions(int len) {

	vector<int> predictions = vector<int>(len);

	for (int i = 0; i < len; i++) {

		vector<float> a = activation[activation.size() - 1].Column(i);

		auto maxElementIterator = std::max_element(a.begin(), a.end());
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

Matrix ReLUDerivative(Matrix total) {
	Matrix a = total;

	for (int r = 0; r < total.RowCount; r++) {
		for (int c = 0; c < total.ColumnCount; c++) {
			a[r][c] = total[r][c] > 0.0f ? 1.0f : 0.0f;
		}
	}
	return a;
}

Matrix SoftMax(Matrix total) {
	return (total - total.LogSumExp()).Exp();
}