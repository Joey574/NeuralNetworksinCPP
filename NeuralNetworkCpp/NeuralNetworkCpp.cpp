#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>
#include <thread>

#include "TestMatrix.h"
using namespace std;

// Hyperparameters
vector<int> dimensions = { 784, 10, 10 };
std::unordered_set<int> resNet = {  };

float learningRate = 0.05;
float thresholdAccuracy = 0.2f;
int batchSize = 500;
int iterations = 250;

// Save / Load
bool SaveOnComplete = false;
bool LoadOnInit = false;
string NetworkPath = "Network.txt";

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
Matrix ELU(Matrix total, float alpha = 1.0f);
Matrix ELUDerivative(Matrix total, float alpha = 1.0f);

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

Matrix SoftMax(Matrix total) {
	return (total - total.LogSumExp()).Exp();
}


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
	YTotal = Matrix(dimensions[dimensions.size() - 1], imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		vector<int> y = vector<int>(dimensions[dimensions.size() - 1], 0);

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

	weights.clear();
	dWeights.clear();
	biases.clear();
	dBiases.clear();

	weights.reserve(dimensions.size() - 1);
	dWeights.reserve(dimensions.size() - 1);

	biases.reserve(dimensions.size() - 1);
	dBiases.reserve(dimensions.size() - 1);

	for (int i = 0; i < dimensions.size() - 1; i++) {
		if (resNet.find(i - 1) != resNet.end()) {
			weights.emplace_back(dimensions[i] + dimensions[0], dimensions[i + 1], -0.5f, 0.5f);
		}
		else {
			weights.emplace_back(dimensions[i], dimensions[i + 1], -0.5f, 0.5f);
		}
		cout << "Weights[" << i << "] connections: " << (weights[i].ColumnCount * weights[i].RowCount) << endl;
		connections += weights[i].ColumnCount * weights[i].RowCount;

		biases.emplace_back(weights[i].Row(0));
		cout << "Biases[" << i << "] connections: " << biases[i].size() << endl;
		connections += biases[i].size();

		dWeights.emplace_back(weights[i].RowCount, weights[i].ColumnCount);
		dBiases.emplace_back(biases[i].size());
	}

	cout << "Total connections: " << connections << endl;

	double fileSize = ((sizeof(float)) * (connections)) + ((sizeof(int) * weights.size() + 1));

	cout << "Predicted size of file: " << (fileSize / 1000000.00) << "mb" << endl;

	InitializeResultMatrices(batchSize);

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
	cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << endl;
}

void InitializeResultMatrices(int size) {
	aTotal.clear();
	activation.clear();
	dTotal.clear();

	aTotal.reserve(weights.size());
	activation.reserve(weights.size());
	dTotal.reserve(weights.size());

	for (int i = 0; i < weights.size(); i++) {
		if (resNet.find(i) != resNet.end()) {
			aTotal.emplace_back(weights[i].ColumnCount + dimensions[0], size);
		}
		else {
			aTotal.emplace_back(weights[i].ColumnCount, size);
		}

		activation.emplace_back(aTotal[i].RowCount, size);
		dTotal.emplace_back(aTotal[i].RowCount, size);
	}
}

Matrix RandomizeInput(Matrix totalInput, int size) {
	Matrix a = Matrix(totalInput.RowCount, size);

	std::unordered_set<int> used = std::unordered_set<int>(size);

	YBatch = Matrix(dimensions[dimensions.size() - 1], size);
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

	int used = 0;

	std::chrono::steady_clock::time_point totalStart;
	std::chrono::steady_clock::time_point totalEnd;

	std::chrono::steady_clock::time_point tStart;
	std::chrono::steady_clock::time_point tEnd;
	std::chrono::duration<double, std::milli> time;

	totalStart = std::chrono::high_resolution_clock::now();

	batch = RandomizeInput(input, batchSize);

	for (int i = 0; i < iterations; i++) {
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

		float acc = Accuracy(GetPredictions(batchSize), batchLabels);

		if (acc > thresholdAccuracy) {
			batch = RandomizeInput(input, batchSize);
		}

		cout << "Iteration: " << i << " Accuracy: " << acc << endl;
	}

	totalEnd = std::chrono::high_resolution_clock::now();

	time = (totalEnd - totalStart);

	float avgTime = time.count() / iterations;

	time /= 1000.00;

	cout << "Total Training Time: " << time.count() << " seconds :: " << (time.count() / 60.00) << " minutes :: " << (time.count() / 3600.00) << " hours" << endl;
	cout << "Average Iteration Time: " << avgTime << " ms" << endl;

}

void ForwardPropogation() {
	/*
	W[0] = 784 x 10
	W[1] = 10 x 10
	B[0] = 10
	B[1] = 10

	A[0] = 10 x 500
	A[1] = 10 x 500

	Z[0] = 10 x 500
	Z[1] = 10 x 500
	
	*/

	std::cout << "Weights[0]: " << weights[0].RowCount << " :: " << weights[0].ColumnCount << std::endl;
	std::cout << "Weights[1]: " << weights[1].RowCount << " :: " << weights[1].ColumnCount << std::endl << std::endl;

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].Transpose().DotProductM(i == 0 ? batch : activation[i - 1]) + biases[i]).Transpose();
		activation[i] = i < aTotal.size() - 1 ? ELU(aTotal[i]) : SoftMax(aTotal[i]);
	}
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - YBatch;

	std::cout << "a" << std::endl;

	for (int i = dTotal.size() - 2; i > -1; i--) {
		std::cout << "DTotal[" << i << "]: " << dTotal[i].RowCount << " :: " << dTotal[i].ColumnCount << std::endl;
		std::cout << "DTotal[" << i + 1 << "]: " << dTotal[i + 1].RowCount << " :: " << dTotal[i + 1].ColumnCount << std::endl;
		std::cout << "Weights[" << i + 1 << "]: " << weights[i + 1].RowCount << " :: " << weights[i + 1].ColumnCount << std::endl;
		std::cout << "ATotal[" << i << "]: " << aTotal[i].RowCount << " :: " << aTotal[i].ColumnCount << std::endl;
		dTotal[i] = ((dTotal[i + 1].DotProductM(weights[i + 1])) * ELUDerivative(aTotal[i]));
	}

	std::cout << std::endl;

	std::cout << "DTotal[0]: " << dTotal[0].RowCount << " :: " << dTotal[0].ColumnCount << std::endl;
	std::cout << "DTotal[1]: " << dTotal[1].RowCount << " :: " << dTotal[1].ColumnCount << std::endl << std::endl;
	std::cout << "Weights[0]: " << weights[0].RowCount << " :: " << weights[0].ColumnCount << std::endl;
	std::cout << "Weights[1]: " << weights[1].RowCount << " :: " << weights[1].ColumnCount << std::endl << std::endl;
	std::cout << "Activation[0]: " << activation[0].RowCount << " :: " << activation[0].ColumnCount << std::endl;
	std::cout << "Activation[1]: " << activation[1].RowCount << " :: " << activation[1].ColumnCount << std::endl;
	std::cout << "Batch: " << batch.RowCount << " :: " << batch.ColumnCount << std::endl << std::endl;

	std::for_each(std::execution::par, dWeights.begin(), dWeights.end(), [&](auto&& item) {
		size_t i = &item - dWeights.data();
		dWeights[i] = (dTotal[i].DotProductM(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize));
		dBiases[i] = dTotal[i].Multiply(1.0f / (float)batchSize).RowSums();
	});

	std::cout << "DWeights[0]: " << dWeights[0].RowCount << " :: " << dWeights[0].ColumnCount << std::endl;
	std::cout << "DWeights[1]: " << dWeights[1].RowCount << " :: " << dWeights[1].ColumnCount << std::endl;

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

	batch = Matrix((width * height), imageNum);
	inputLabels = vector<int>(imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		testingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		batch.SetColumn(i, intData);

		char byte;
		testingLabelFR.read(&byte, 1);
		inputLabels[i] = static_cast<int>(static_cast<unsigned char>(byte));
	}
	testingFR.close();
	testingLabelFR.close();

	batch = batch.Divide(255);

	InitializeResultMatrices(batch.ColumnCount);

	ForwardPropogation();

	float acc = Accuracy(GetPredictions(batch.ColumnCount), inputLabels);

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