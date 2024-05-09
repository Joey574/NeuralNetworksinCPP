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
std::vector<int> vnn_dimensions = { 784, 16, 16, 1 };
std::vector<int> gpnn_dimensions = { 784, 32, 32, 10 };
std::vector<int> dnn_dimensions = { 784, 32, 32, 10 };
std::unordered_set<int> resNet = {  };
int fourierSeries = 0;
int taylorSeries = 0;

float lowerNormalized = 0.0f;
float upperNormalized = 1.0f;

Matrix::init initType = Matrix::init::He;
int epochs = 100;
int batchSize = 500;
float learningRate = 0.05;

// Inputs
Matrix input;
Matrix testData;

std::vector<int> inputLabels;
std::vector<int> testLabels;

// Neural Network Matrices
std::vector<Matrix> weights;
std::vector<std::vector<float>> biases;

// Error stuff
Matrix YTotal;
Matrix YBatch;

// Prototypes
void LoadInput();
int ReadBigInt(std::ifstream* fr);
std::tuple<Matrix, Matrix, std::vector<int>> ShuffleInput(Matrix in, Matrix Y, std::vector<int> labels);
std::tuple<Matrix, Matrix, std::vector<int>> MakeDataset(Matrix data, Matrix Y, std::vector<int> labels, int num);

int main()
{
	LoadInput();
	MakeDataset(input, YTotal, inputLabels, 1);
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


std::tuple<Matrix, Matrix, std::vector<int>> MakeDataset(Matrix data, Matrix Y, std::vector<int> labels, int num) {

	// Find all instances of num in data
	std::vector<int> target_index;
	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == num) {
			target_index.push_back(i);
		}
	}

	std::unordered_set<int> used (target_index.begin(), target_index.end());
	std::vector<int> value_index;

	// Get random instances that aren't num equal to size of instances of num
	for (int i = 0; value_index.size() < target_index.size(); i++) {

		int r = rand() % labels.size();

		if (used.find(r) == used.end()) {
			used.insert(r);

			value_index.push_back(r);
		}
	}

	// Get all indexes used
	std::vector<int> all_indexes;

	all_indexes.reserve(used.size());
	all_indexes.insert(all_indexes.end(), target_index.begin(), target_index.end());
	all_indexes.insert(all_indexes.end(), value_index.begin(), value_index.end());

	Matrix dataset = Matrix(data.RowCount, all_indexes.size());
	Matrix dataset_y = Matrix(Y.RowCount, all_indexes.size());
	std::vector<int> dataset_labels;

	// Actually make dataset
	for (int i = 0; i < all_indexes.size(); i++) {
		dataset.SetColumn(i, data.Column(i));
		dataset_y.SetColumn(i, Y.Column(i));
		dataset_labels.push_back(labels[i]);
	}

	return std::make_tuple(dataset, dataset_y, dataset_labels);
}



void TrainVNN() {

	Matrix VNN_Data;
	Matrix VNN_Y;
	std::vector<int> VNN_Labels;

	for (int v = 0; v < 10; v++) {

		// Make dataset containing 50% target value
		std::tie(VNN_Data, VNN_Y, VNN_Labels) = MakeDataset(input, YTotal, inputLabels, v);

		int iterations = VNN_Data.ColumnCount / batchSize;

		for (int e = 0; e < epochs; e++) {

			std::tie(VNN_Data, VNN_Y, VNN_Labels) = ShuffleInput(VNN_Data, VNN_Y, VNN_Labels);

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

std::tuple<std::vector<Matrix>, std::vector<std::vector<float>> > BackwardPropogation(Matrix in, Matrix Y, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res) {

	std::vector<Matrix> dT = std::vector<Matrix>();
	std::vector<Matrix> dW = std::vector<Matrix>();
	std::vector<std::vector<float>> dB = std::vector<std::vector<float>>();
	
	for (int i = 0; i < w.size(); i++) {
		dT.emplace_back(A[i].RowCount, A[i].ColumnCount);

		dW.emplace_back(w[i].RowCount, w[i].ColumnCount);
		dB.emplace_back(b[i].size());
	}

	// Backward prop
	dT[dT.size() - 1] = A[A.size() - 1] - Y;

	for (int i = dT.size() - 2; i > -1; i--) {

		if (res.find(i) != res.end()) {
			dT[i] = ((dT[i + 1].DotProduct(w[i + 1].SegmentR(in.RowCount))).Transpose() * LeakyReLUDerivative(Z[i].SegmentR(in.RowCount)));
		}
		else {
			dT[i] = ((dT[i + 1].DotProduct(w[i + 1])).Transpose() * LeakyReLUDerivative(Z[i]));
		}
	}

	std::for_each(std::execution::par, dW.begin(), dW.end(), [&](auto&& item) {
		size_t i = &item - dW.data();
		item = (dT[i].Transpose().DotProduct(i == 0 ? in.Transpose() : A[i - 1].Transpose()) * (1.0f / (float)in.ColumnCount)).Transpose();
		dB[i] = dT[i].Multiply(1.0f / (float)in.ColumnCount).RowSums();
	});

	// Update Network
	for (int i = 0; i < w.size(); i++) {
		w[i] -= dW[i].Multiply(learningRate);
	}

	for (int i = 0; i < b.size(); i++) {
		for (int x = 0; x < b[i].size(); x++) {
			b[i][x] -= (dB[i][x] * learningRate);
		}
	}

	return std::make_tuple(w, b);
}