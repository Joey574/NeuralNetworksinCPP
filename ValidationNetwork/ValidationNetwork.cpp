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
#include <tuple>

#include "Matrix.h"
#include "ActivationFunctions.h"

// Hyperparameters
std::vector<int> dimensions = { 784, 16, 16, 10 };
std::unordered_set<int> resNet = {  };
int fourierSeries = 0;
int taylorSeries = 0;

float lowerNormalized = 0.0f;
float upperNormalized = 1.0f;

Matrix::init initType = Matrix::init::He;
int epochs = 100;
int batchSize = 500;
float learningRate = 0.05;

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
void LoadInput();
int ReadBigInt(std::ifstream* fr);
std::tuple<Matrix, Matrix, std::vector<int>> ShuffleInput(Matrix in, Matrix Y, std::vector<int> labels);

int main()
{
	LoadInput();
}


std::tuple<Matrix, Matrix, std::vector<int>> ShuffleInput(Matrix in, Matrix Y, std::vector<int> labels) {

	// Shuffle input
	for (int k = 0; k < in.ColumnCount; k++) {

		int r = k + rand() % (in.ColumnCount - k);

		std::vector<float> tempI = in.Column(k);
		std::vector<float> tempY = Y.Column(k);
		int tempL = labels[k];

		in.SetColumn(k, in.Column(r));
		Y.SetColumn(k, Y.Column(r));
		labels[k] = labels[r];

		in.SetColumn(r, tempI);
		Y.SetColumn(r, tempY);
		labels[r] = tempL;
	}

	return std::make_tuple( in, Y, labels );
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

	input = input.Normalized(lowerNormalized, upperNormalized);
	testData = testData.Normalized(lowerNormalized, upperNormalized);

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


void TrainVNN() {

	int iterations = input.ColumnCount / batchSize;

	Matrix VNNData;
	Matrix VNNY;
	std::vector<int> VNNLabels;

	for (int v = 0; v < 10; v++) {

		// TODO Make dataset -> want 50% target value + 50% random spread of other

		for (int e = 0; e < epochs; e++) {

			std::tie(VNNData, VNNY, VNNLabels) = ShuffleInput(VNNData, VNNY, VNNLabels);

			for (int i = 0; i < iterations; i++) {

				// std::tie( *vnnA, vnnZ) = ForwardPropogation(VNNData, VNNWeights[v], VNNBiases[v], VNNActivation[v], VNNATotal[v], renet[v]);

			}
		}
	}
}

void TrainGPNN() {

}

void TrainDNN() {

}

std::tuple<std::vector<Matrix>, std::vector<Matrix>> ForwardPropogation(Matrix in, std::vector<Matrix> w, std::vector<std::vector<float>> b, 
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res) {

	for (int i = 0; i < Z.size(); i++) {
		if (res.find(i) != res.end()) {

			Z[i].Insert(0, in);
			A[i].Insert(0, in);

			Z[i].Insert(in.RowCount, (w[i].DotProduct(i == 0 ? in : A[i - 1]) + b[i]).Transpose());
		}
		else {
			Z[i] = (w[i].DotProduct(i == 0 ? in : A[i - 1]) + b[i]).Transpose();
		}
		A[i] = i < Z.size() - 1 ? LeakyReLU(Z[i]) : SoftMax(Z[i]);
	}

	return std::make_tuple( A, Z );
}