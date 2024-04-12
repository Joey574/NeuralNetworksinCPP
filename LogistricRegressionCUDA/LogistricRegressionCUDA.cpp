#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

#define _USE_MATH_DEFINES
#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>
#include <thread>
#include <iomanip>
#include <cmath>

#include "Matrix.h"

//Hyperparameters
std::vector<int> dimensions = { 784, 16, 16, 10 };
std::unordered_set<int> resNet = {  };
int fourierSeries = 0;
int taylorSeries = 0;

float lowerNormalized = 0;
float upperNormalized = 1.0;

Matrix::init initType = Matrix::init::He;
int epochs = 25;
int batchSize = 500;
float learningRate = 0.1;

// Save / Load
bool SaveOnComplete = false;
bool LoadOnInit = false;
std::string NetworkPath = "Network.txt";

// Inputs
Matrix input;
Matrix testData;
Matrix batch;

std::vector<int> inputLabels;
std::vector<int> testLabels;
std::vector<int> batchLabels;

// Neural Network Matrices
std::vector<Matrix> weights;
std::vector<std::vector<float>> biases;

std::vector<Matrix> activation;
std::vector<Matrix> aTotal;

std::vector<Matrix> dTotal;
std::vector<Matrix> dWeights;
std::vector<std::vector<float>> dBiases;

// Error stuff
Matrix YTotal;
Matrix YBatch;

// Prototypes
void InitializeNetwork();
void InitializeResultMatrices(int size);
void TrainNetwork();
void ForwardPropogation(Matrix in);
void BackwardPropogation();
void UpdateNetwork();
void LoadInput();
int ReadBigInt(std::ifstream* fr);
Matrix GetNextInput(Matrix totalInput, int size, int i);
std::vector<int> GetPredictions(int len);
float Accuracy(std::vector<int> predictions, std::vector<int> labels);
void SaveNetwork(std::string filename);
void LoadNetwork(std::string filename);
void CleanTime(double time);
void ShuffleInput();
Matrix SoftMax(Matrix total);

int main()
{
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	srand(time(0));

	LoadInput();

	if (LoadOnInit) {
		LoadNetwork(NetworkPath);
	}
	else {
		InitializeNetwork();
	}

	TrainNetwork();

	if (SaveOnComplete) { SaveNetwork(NetworkPath); }

	return 0;
}

Matrix SoftMax(Matrix total) {
	return (total - total.LogSumExp()).Exp();
}

void ShuffleInput() {
	// Shuffle input
	for (int k = 0; k < input.ColumnCount; k++) {

		int r = k + rand() % (input.ColumnCount - k);

		std::vector<float> tempI = input.Column(k);
		std::vector<float> tempY = YTotal.Column(k);
		int tempL = inputLabels[k];

		input.SetColumn(k, input.Column(r));
		YTotal.SetColumn(k, YTotal.Column(r));
		inputLabels[k] = inputLabels[r];

		input.SetColumn(r, tempI);
		YTotal.SetColumn(r, tempY);
		inputLabels[r] = tempL;
	}
}

void LoadInput() {

	auto sTime = std::chrono::high_resolution_clock::now();

	// Train Data
	std::string trainingImages = "Training Data\\train-images.idx3-ubyte";
	std::string trainingLabels = "Training Data\\train-labels.idx1-ubyte";

	std::ifstream trainingFR = std::ifstream(trainingImages, std::ios::binary);
	std::ifstream trainingLabelsFR = std::ifstream(trainingLabels, std::ios::binary);

	if (trainingFR.is_open() && trainingLabelsFR.is_open()) {
		std::cout << "Loading training data..." << std::endl;
	}
	else {
		std::cout << "File(s) not found" << std::endl;
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
	inputLabels = std::vector<int>(imageNum);
	YTotal = Matrix(dimensions[dimensions.size() - 1], imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		std::vector<int> y = std::vector<int>(dimensions[dimensions.size() - 1], 0);

		char byte;
		trainingLabelsFR.read(&byte, 1);
		int label = static_cast<int>(static_cast<unsigned char>(byte));

		inputLabels[i] = label;

		y[label] = 1;
		YTotal.SetColumn(i, y);
	}

	trainingFR.close();
	trainingLabelsFR.close();

	// Test Data
	std::string testingImages = "Testing Data\\t10k-images.idx3-ubyte";
	std::string testingLabels = "Testing Data\\t10k-labels.idx1-ubyte";

	std::ifstream testingFR = std::ifstream(testingImages, std::ios::binary);
	std::ifstream testingLabelFR = std::ifstream(testingLabels, std::ios::binary);

	if (testingFR.is_open() && testingLabelFR.is_open()) {
		std::cout << "Loading testing data..." << std::endl;
	}
	else {
		std::cout << "File(s) not found" << std::endl;
	}

	// Discard
	magicNum = ReadBigInt(&testingLabelFR);
	imageNum = ReadBigInt(&testingLabelFR);
	magicNum = ReadBigInt(&testingFR);

	// Read the important things
	imageNum = ReadBigInt(&testingFR);
	width = ReadBigInt(&testingFR);
	height = ReadBigInt(&testingFR);

	testData = Matrix((width * height), imageNum);
	testLabels = std::vector<int>(imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		testingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		testData.SetColumn(i, intData);

		char byte;
		testingLabelFR.read(&byte, 1);
		testLabels[i] = static_cast<int>(static_cast<unsigned char>(byte));
	}
	testingFR.close();
	testingLabelFR.close();

	input = input.NormalizeTo(lowerNormalized, upperNormalized);
	testData = testData.NormalizeTo(lowerNormalized, upperNormalized);

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - sTime;
	std::cout << "Time to load input: " << (time.count() / 1000.00) << " seconds" << std::endl;

	Matrix oldI = input;
	Matrix oldT = testData;

	// Compute Fourier Series
	if (fourierSeries > 0) {
		sTime = std::chrono::high_resolution_clock::now();
		std::cout << "Computing " << fourierSeries << " order(s) of Fourier Series..." << std::endl;

		for (int f = 0; f < fourierSeries; f++) {
			input = input.Combine(oldI.FourierSeries(f + 1));
			testData = testData.Combine(oldT.FourierSeries(f + 1));
		}
		dimensions[0] = input.RowCount;

		std::cout << "Fourier Features: " << input.RowCount - oldI.RowCount << std::endl;

		time = std::chrono::high_resolution_clock::now() - sTime;
		std::cout << "Time to compute " << fourierSeries << " order(s): " << time.count() / 1000.00 << " seconds" << std::endl;
	}

	// Compute Taylor Series
	if (taylorSeries > 0) {
		sTime = std::chrono::high_resolution_clock::now();
		std::cout << "Computing " << taylorSeries << " order(s) of Taylor Series..." << std::endl;

		for (int t = 1; t < taylorSeries + 1; t++) {
			input = input.Combine(oldI.TaylorSeries(t + 1));
			testData = testData.Combine(oldT.TaylorSeries(t + 1));
		}
		dimensions[0] = input.RowCount;

		time = std::chrono::high_resolution_clock::now() - sTime;
		std::cout << "Time to compute " << taylorSeries << " order(s): " << time.count() / 1000.00 << " seconds" << std::endl;
	}
}

int ReadBigInt(std::ifstream* fr) {

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
			weights.emplace_back(dimensions[i] + dimensions[0], dimensions[i + 1], initType);
		}
		else {
			weights.emplace_back(dimensions[i], dimensions[i + 1], initType);
		}
		std::cout << "Weights[" << i << "] connections: " << (weights[i].ColumnCount * weights[i].RowCount) << std::endl;
		connections += weights[i].ColumnCount * weights[i].RowCount;

		biases.emplace_back(std::vector<float>(dimensions[i + 1], 0));

		std::cout << "Biases[" << i << "] connections: " << biases[i].size() << std::endl;
		connections += biases[i].size();

		dWeights.emplace_back(weights[i].RowCount, weights[i].ColumnCount);
		dBiases.emplace_back(biases[i].size());
	}

	double fileSize = ((sizeof(float)) * (connections)) + ((sizeof(int) * weights.size() + 1));

	std::cout << "Total connections: " << connections << std::endl;
	std::cout << "Predicted size of file: " << (fileSize / 1000000.00) << "mb" << std::endl;

	aTotal.reserve(weights.size());
	activation.reserve(weights.size());
	dTotal.reserve(weights.size());

	for (int i = 0; i < weights.size(); i++) {
		if (resNet.find(i) != resNet.end()) {
			aTotal.emplace_back(weights[i].ColumnCount + dimensions[0], batchSize);
		}
		else {
			aTotal.emplace_back(weights[i].ColumnCount, batchSize);
		}
		activation.emplace_back(aTotal[i].RowCount, batchSize);
		dTotal.emplace_back(aTotal[i].RowCount, batchSize);
	}

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
	std::cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << std::endl;
}

void InitializeResultMatrices(int size) {
	aTotal.clear();
	activation.clear();

	aTotal.reserve(weights.size());
	activation.reserve(weights.size());

	for (int i = 0; i < weights.size(); i++) {
		if (resNet.find(i) != resNet.end()) {
			aTotal.emplace_back(weights[i].ColumnCount + dimensions[0], size);
		}
		else {
			aTotal.emplace_back(weights[i].ColumnCount, size);
		}

		activation.emplace_back(aTotal[i].RowCount, size);
	}
}

Matrix GetNextInput(Matrix totalInput, int size, int i) {
	Matrix a = Matrix(totalInput.RowCount, size);

	YBatch = YTotal.SegmentC(i * size, i * size + size);
	a = totalInput.SegmentC(i * size, i * size + size);
	batchLabels.clear();

	for (int x = i * size; x < i * size + size; x++) {
		batchLabels.push_back(inputLabels[x]);
	}

	return a;
}

void TrainNetwork() {
	std::cout << "TRAINING STARTED" << std::endl;

	std::chrono::steady_clock::time_point totalStart;
	std::chrono::steady_clock::time_point tStart;
	std::chrono::duration<double, std::milli> time;
	std::chrono::duration<double, std::milli> timeToReachHighest;

	totalStart = std::chrono::high_resolution_clock::now();

	int iterations = input.ColumnCount / batchSize;

	float highestAcc = 0.0f;
	int highestIndex = 0;

	std::cout << std::fixed << std::setprecision(4);
	for (int e = 0; e < epochs; e++) {

		tStart = std::chrono::high_resolution_clock::now();

		ShuffleInput();
		for (int i = 0; i < iterations; i++) {

			batch = GetNextInput(input, batchSize, i);

			ForwardPropogation(batch);
			BackwardPropogation();
			UpdateNetwork();
		}

		InitializeResultMatrices(testData.ColumnCount);
		ForwardPropogation(testData);
		float acc = Accuracy(GetPredictions(testData.ColumnCount), testLabels);

		if (acc >= highestAcc) {
			highestAcc = acc;
			highestIndex = e;
			timeToReachHighest = std::chrono::high_resolution_clock::now() - totalStart;
		}

		InitializeResultMatrices(batchSize);

		time = std::chrono::high_resolution_clock::now() - tStart;
		std::cout << "Epoch: " << e << " Accuracy: " << acc << " Epoch Time: ";
		CleanTime(time.count());
	}

	time = (std::chrono::high_resolution_clock::now() - totalStart);
	float epochTime = time.count() / epochs;

	std::cout << "Total Training Time: ";
	CleanTime(time.count());
	std::cout << "Average Epoch Time: ";
	CleanTime(epochTime);

	std::cout << "Highest Accuracy: " << highestAcc << " at epoch " << highestIndex << std::endl;
	std::cout << "Time to reach max: ";
	CleanTime(timeToReachHighest.count());
}

void ForwardPropogation(Matrix in) {

	for (int i = 0; i < aTotal.size(); i++) {
		if (resNet.find(i) != resNet.end()) {

			aTotal[i].Insert(0, in);
			activation[i].Insert(0, in);

			aTotal[i].Insert(in.RowCount, (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose());
		}
		else {
			aTotal[i] = (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose();
		}
		activation[i] = i < aTotal.size() - 1 ? (aTotal[i].LeakyReLU()) : SoftMax(aTotal[i]);
	}
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - YBatch;

	for (int i = dTotal.size() - 2; i > -1; i--) {

		if (resNet.find(i) != resNet.end()) {
			dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1].SegmentR(batch.RowCount))).Transpose() * (aTotal[i].SegmentR(batch.RowCount).LeakyReLUDeriv()));
		}
		else {
			dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])).Transpose() * (aTotal[i].LeakyReLUDeriv()));
		}
	}

	std::for_each(std::execution::par_unseq, dWeights.begin(), dWeights.end(), [&](auto&& item) {
		size_t i = &item - dWeights.data();
		item = (dTotal[i].Transpose().DotProduct(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize)).Transpose();
		dBiases[i] = dTotal[i].Multiply(1.0f / (float)batchSize).RowSums();
		});

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

std::vector<int> GetPredictions(int len) {

	std::vector<int> predictions = std::vector<int>(len);

	for (int i = 0; i < len; i++) {

		std::vector<float> a = activation[activation.size() - 1].Column(i);

		auto maxElementIterator = std::max_element(a.begin(), a.end());
		predictions[i] = std::distance(a.begin(), maxElementIterator);
	}
	return predictions;
}

float Accuracy(std::vector<int> predictions, std::vector<int> labels) {
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

void SaveNetwork(std::string filename) {
	std::ofstream fw = std::ofstream(filename, std::ios::out | std::ios::binary);

	int s = weights.size() - 1;
	fw.write(reinterpret_cast<const char*>(&s), sizeof(int));

	std::vector<int> dims = std::vector<int>(dimensions.size());
	for (int i = 0; i < dims.size(); i++) {
		dims[i] = dimensions[i];
	}

	fw.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(int));

	for (int i = 0; i < weights.size(); i++) {
		for (int r = 0; r < weights[i].RowCount; r++) {
			fw.write(reinterpret_cast<const char*>(weights[i].Row(r).data()), weights[i].Row(r).size() * sizeof(float));
		}
	}

	for (int i = 0; i < biases.size(); i++) {
		fw.write(reinterpret_cast<const char*>(biases[i].data()), biases[i].size() * sizeof(float));
	}

	std::cout << "NETWORK SAVED" << std::endl;

	fw.close();
}

void LoadNetwork(std::string filename) {
	std::ifstream fr = std::ifstream(filename, std::ios::in | std::ios::binary);

	if (fr.is_open()) {
		std::cout << "Loading Network..." << std::endl;
	}
	else {
		std::cout << "Network not found..." << std::endl;
	}

	int s;
	fr.read(reinterpret_cast<char*>(&s), sizeof(int));

	dimensions = std::vector<int>(s);
	fr.read(reinterpret_cast<char*>(dimensions.data()), s * sizeof(int));

	InitializeNetwork();

	for (int i = 0; i < weights.size(); i++) {
		for (int r = 0; r < weights[i].RowCount; r++) {
			std::vector<float> row = std::vector<float>(weights[i].ColumnCount);
			fr.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

			weights[i].SetRow(r, row);
		}
	}

	for (int i = 0; i < biases.size(); i++) {
		fr.read(reinterpret_cast<char*>(biases[i].data()), biases[i].size() * sizeof(float));
	}

	fr.close();

	std::cout << "NETWORK LOADED" << std::endl;
}

void CleanTime(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;

	if (time / hour > 1.00) {
		std::cout << time / hour << " hours";
	}
	else if (time / minute > 1.00) {
		std::cout << time / minute << " minutes";
	}
	else if (time / second > 1.00) {
		std::cout << time / second << " seconds";
	}
	else {
		std::cout << time << " ms";
	}
	std::cout << std::endl;
}