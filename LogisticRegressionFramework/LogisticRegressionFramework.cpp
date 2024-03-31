#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>
#include <sstream>
#include <thread>

#include "NetworkFramework.h"
#include "Matrix.h"

using namespace std;

// Hyperparameters
vector<int> dimensions = { 784, 128, 10 };

float learningRate = 0.05f;
float thresholdAccuracy = 0.2f;
int batchSize = 500;
int iterations = 250;

// Inputs
Matrix input;
vector<int> inputLabels;

// Test data
Matrix testData;
vector<int> testLabels;

// Error stuff
Matrix YTotal;


// Testing
std::vector<NetworkFramework> testNetworks;
Matrix accuracy;


void LoadTestData();
void LoadInput();
int ReadBigInt(ifstream* fr);

int main()
{
	LoadInput();
	LoadTestData();

	testNetworks = std::vector<NetworkFramework>(10);

	for (int i = 0; i < testNetworks.size(); i++) {
		testNetworks[i].SetNetworkDimensions(dimensions);
		testNetworks[i].SetInput(input, inputLabels, YTotal);
		testNetworks[i].SetTestData(testData, testLabels);
		testNetworks[i].SetHyperparameters(learningRate, thresholdAccuracy, batchSize, iterations);
		testNetworks[i].InitializeNetwork();
		testNetworks[i].SetActivationFunctions(ELU, ELUDerivative, (float)i + 1.0f);
	}

	accuracy = Matrix(testNetworks.size(), 9);

	for (int i = 0; i < 9; i++) {

		std::cout << "Iteration: " << i << std::endl;
		std::vector<int> a = std::vector<int>(testNetworks.size());

		std::for_each(std::execution::par, testNetworks.begin(), testNetworks.end(), [&](auto&& item) {
			size_t i = &item - testNetworks.data();

			item.TrainNetwork();
			float a = item.TestNetwork();
			std::stringstream ss;
			ss << (i + 1) << ": " << a << "\n";
			std::cout << ss.str();
			testNetworks[i].InitializeNetwork();
		});
	}
	return 0;
}

void LoadTestData() {
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

	testData = testData.Divide(255);
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
	YTotal = Matrix((width * height), imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		vector<int> y = vector<int>((width * height), 0);

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
