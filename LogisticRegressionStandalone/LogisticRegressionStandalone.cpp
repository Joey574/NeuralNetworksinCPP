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
#include "ActivationFunctions.h"

using namespace std;

// Hyperparameters
vector<int> dimensions = { 784, 16, 16, 10 };
std::unordered_set<int> resNet = {  };
int fourierSeries = 1;

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

Matrix::init initType = Matrix::init::He;
int epochs = 35;
int batchSize = 250;
float learningRate = 0.05;

// Save / Load
bool SaveOnComplete = false;
bool LoadOnInit = false;
string NetworkPath = "Network.txt";

// Inputs
Matrix input;
Matrix testData;
Matrix batch;

vector<int> inputLabels;
vector<int> testLabels;
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
void ForwardPropogation(Matrix in);
void BackwardPropogation();
void UpdateNetwork();
void LoadInput();
int ReadBigInt(ifstream* fr);
Matrix GetNextInput(Matrix totalInput, int size, int i);
vector<int> GetPredictions(int len);
float Accuracy(vector<int> predictions, vector<int> labels);
void SaveNetwork(string filename);
void LoadNetwork(string filename);

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

	TestNetwork();

	if (SaveOnComplete) { SaveNetwork(NetworkPath); }

	return 0;
}

void LoadInput() {

	auto sTime = std::chrono::high_resolution_clock::now();

	// Train Data
	string trainingImages = "Training Data\\train-images.idx3-ubyte";
	string trainingLabels = "Training Data\\train-labels.idx1-ubyte";

	ifstream trainingFR = ifstream(trainingImages, std::ios::binary);
	ifstream trainingLabelsFR = ifstream(trainingLabels, std::ios::binary);

	if (trainingFR.is_open() && trainingLabelsFR.is_open()) {
		std::cout << "Loading training data..." << endl;
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

	// Test Data
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
	magicNum = ReadBigInt(&testingLabelFR);
	imageNum = ReadBigInt(&testingLabelFR);
	magicNum = ReadBigInt(&testingFR);

	// Read the important things
	imageNum = ReadBigInt(&testingFR);
	width = ReadBigInt(&testingFR);
	height = ReadBigInt(&testingFR);

	testData = Matrix((width * height), imageNum);
	testLabels = vector<int>(imageNum);

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

	input = input.Normalized(lowerNormalized, upperNormalized);
	testData = testData.Normalized(lowerNormalized, upperNormalized);

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - sTime;
	std::cout << "Time to load input: " << (time.count() / 1000.00) << " seconds" << std::endl;

	// Compute Fourier Series
	if (fourierSeries > 0) {
		sTime = std::chrono::high_resolution_clock::now();

		std::cout << "Computing " << fourierSeries << " order(s) of Fourier Series..." << std::endl;

		Matrix oldI = input;
		Matrix oldT = testData;
		for (int f = 0; f < fourierSeries; f++) {
			input = input.Combine(oldI.FourierSeries(f + 1));
			testData = testData.Combine(oldT.FourierSeries(f + 1));
		}
		dimensions[0] = input.RowCount;

		time = std::chrono::high_resolution_clock::now() - sTime;
		std::cout << "Time to compute " << fourierSeries << " order(s): " << time.count() / 1000.00 << " seconds" << std::endl;
	}
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
			weights.emplace_back(dimensions[i] + dimensions[0], dimensions[i + 1], initType);
		} else {
			weights.emplace_back(dimensions[i], dimensions[i + 1], initType);
		}
		cout << "Weights[" << i << "] connections: " << (weights[i].ColumnCount * weights[i].RowCount) << endl;
		connections += weights[i].ColumnCount * weights[i].RowCount;

		biases.emplace_back(vector<float>(dimensions[i + 1], 0));

		cout << "Biases[" << i << "] connections: " << biases[i].size() << endl;
		connections += biases[i].size();

		dWeights.emplace_back(weights[i].RowCount, weights[i].ColumnCount);
		dBiases.emplace_back(biases[i].size());
	}

	double fileSize = ((sizeof(float)) * (connections)) + ((sizeof(int) * weights.size() + 1));

	cout << "Total connections: " << connections << endl;
	cout << "Predicted size of file: " << (fileSize / 1000000.00) << "mb" << endl;

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
	cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << endl;
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

	std::cout << "TRAINING STARTED" << endl;

	std::chrono::steady_clock::time_point totalStart;
	std::chrono::steady_clock::time_point tStart;
	std::chrono::duration<double, std::milli> time;

	totalStart = std::chrono::high_resolution_clock::now();

	int iterations = input.ColumnCount / batchSize;
	float highestAcc = 0.0f;
	int highestIndex = 0;

	for (int e = 0; e < epochs; e++) {

		tStart = std::chrono::high_resolution_clock::now();
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
		}

		InitializeResultMatrices(batchSize);

		time = std::chrono::high_resolution_clock::now() - tStart;
		std::cout << "Epoch: " << e << " Accuracy: " << std::fixed << std::setprecision(3) << acc << 
			" Epoch Time: " << time.count() << " ms :: " << time.count() / 1000.00 << " seconds :: " << time.count() / 60000.00 << " minutes" << std::endl;
	}

	time = (std::chrono::high_resolution_clock::now() - totalStart) / 1000.00;
	float epochTime = time.count() / epochs;

	std::cout << "Total Training Time: " << time.count() << " seconds :: " << (time.count() / 60.00) << " minutes :: " << (time.count() / 3600.00) << " hours" << std::endl;
	std::cout << "Average Epoch Time: " << epochTime << " seconds :: " << (epochTime / 60.00) << " minutes" << std::endl;
	std::cout << "Highest Accuracy: " << highestAcc << " at epoch " << highestIndex << std::endl;
}

void ForwardPropogation(Matrix in) {

	for (int i = 0; i < aTotal.size(); i++) {
		if (resNet.find(i) != resNet.end()) {
			
			aTotal[i].Insert(0, in);
			activation[i].Insert(0, in);

			aTotal[i].Insert(in.RowCount, (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose());
		} else {
			aTotal[i] = (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose();
		}
		activation[i] = i < aTotal.size() - 1 ? LeakyReLU(aTotal[i]) : SoftMax(aTotal[i]);
	}
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - YBatch;

	for (int i = dTotal.size() - 2; i > -1; i--) {

		if (resNet.find(i) != resNet.end()) {
			dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1].SegmentR(batch.RowCount))).Transpose() * LeakyReLUDerivative(aTotal[i].SegmentR(batch.RowCount)));
		}
		else {
			dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])).Transpose() * LeakyReLUDerivative(aTotal[i]));
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

void TestNetwork() {
	InitializeResultMatrices(testData.ColumnCount);
	ForwardPropogation(testData);

	float acc = Accuracy(GetPredictions(testData.ColumnCount), testLabels);
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

void SaveNetwork(string filename) {
	ofstream fw = ofstream(filename, ios::out | ios::binary);

	int s = weights.size() - 1;
	fw.write(reinterpret_cast<const char*>(&s), sizeof(int));

	vector<int> dims = vector<int>(dimensions.size());
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

	cout << "NETWORK SAVED" << endl;

	fw.close();
}

void LoadNetwork(string filename) {
	ifstream fr = ifstream(filename, ios::in | ios::binary);

	if (fr.is_open()) {
		cout << "Loading Network..." << endl;
	}
	else {
		cout << "Network not found..." << endl;
	}

	int s;
	fr.read(reinterpret_cast<char*>(&s), sizeof(int));

	dimensions = vector<int>(s);
	fr.read(reinterpret_cast<char*>(dimensions.data()), s * sizeof(int));

	InitializeNetwork();

	for (int i = 0; i < weights.size(); i++) {
		for (int r = 0; r < weights[i].RowCount; r++) {
			vector<float> row = vector<float>(weights[i].ColumnCount);
			fr.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

			weights[i].SetRow(r, row);
		}
	}

	for (int i = 0; i < biases.size(); i++) {
		fr.read(reinterpret_cast<char*>(biases[i].data()), biases[i].size() * sizeof(float));
	}

	fr.close();

	cout << "NETWORK LOADED" << endl;
}