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
std::vector<int> vnn_dimensions = { 784, 30, 30, 1 };
std::vector<int> gpnn_dimensions = { 784, 32, 32, 10 };
std::vector<int> dnn_dimensions = { 784, 32, 32, 10 };

std::unordered_set<int> vnn_res = {  }; 

float lowerNormalized = 0.0f;
float upperNormalized = 1.0f;

Matrix::init initType = Matrix::init::He;
int epochs = 50;
int batchSize = 500;
float learningRate = 0.01;

// Feature Engineering
int fourierSeries = 0;
int chebyshevSeries = 0;
int taylorSeries = 0;
int legendreSeries = 0;
int laguerreSeries = 0;

// Inputs
Matrix input;
Matrix testData;
Matrix YTotal;

std::vector<float> inputLabels;
std::vector<float> testLabels;

// VNN weights and biases
std::vector<std::vector<Matrix>> vnn_weights;
std::vector<std::vector<std::vector<float>>> vnn_biases;

// VNN results
std::vector<std::vector<Matrix>> vnn_total;
std::vector<std::vector<Matrix>> vnn_activation;

// Prototypes
void CleanTime(double time);
void InitializeNetworks();
std::tuple<Matrix, std::vector<float>> ShuffleInput(Matrix in, std::vector<float> labels);
void LoadInput();
int ReadBigInt(std::ifstream* fr);

std::tuple<Matrix, std::vector<float>> MakeDataset(Matrix data, std::vector<float> labels, int num);
std::vector<float> MakeTestDataset(std::vector<float> labels, int num);
std::tuple<Matrix, std::vector<float>> GetNextInput(Matrix in, std::vector<float> labels, int size, int i);
void TrainVNN();
void TrainGPNN();
void TrainDNN();
std::tuple<std::vector<Matrix>, std::vector<Matrix>> ForwardPropogation(Matrix in, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res);
std::tuple<std::vector<Matrix>, std::vector<std::vector<float>> > BackwardPropogation(Matrix in, std::vector<float> labels, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res);
float TestNetwork(std::vector<Matrix> w, std::vector<std::vector<float>> b, std::unordered_set<int> res, std::vector<float> labels);


int main()
{
	LoadInput();

	InitializeNetworks();

	TrainVNN();
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

void InitializeNetworks() {
	auto initStart = std::chrono::high_resolution_clock::now();

	vnn_weights.clear();
	vnn_biases.clear();

	vnn_weights.reserve(10);
	vnn_biases.reserve(10);

	vnn_total.reserve(10);
	vnn_activation.reserve(10);

	std::vector<Matrix> weight;
	std::vector<std::vector<float>> bias;

	std::vector<Matrix> total;
	std::vector<Matrix> activation;

	for (int v = 0; v < 10; v++) {

		weight.clear();
		bias.clear();

		total.clear();
		activation.clear();

		for (int i = 0; i < vnn_dimensions.size() - 1; i++) {

			if (vnn_res.find(i - 1) != vnn_res.end()) {
				weight.emplace_back(vnn_dimensions[i] + vnn_dimensions[0], vnn_dimensions[i + 1], initType);
			}
			else {
				weight.emplace_back(vnn_dimensions[i], vnn_dimensions[i + 1], initType);
			}
			bias.emplace_back(std::vector<float>(vnn_dimensions[i + 1], 0));

		}
		
		for (int i = 0; i < weight.size(); i++) {
			if (vnn_res.find(i) != vnn_res.end()) {
				total.emplace_back(weight[i].ColumnCount + vnn_dimensions[0], batchSize);
			}
			else {
				total.emplace_back(weight[i].ColumnCount, batchSize);
			}
			activation.emplace_back(total[i].RowCount, batchSize);
		}

		vnn_weights.push_back(weight);
		vnn_biases.push_back(bias);

		vnn_total.push_back(total);
		vnn_activation.push_back(activation);
	}

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
	std::cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << std::endl;
}

std::tuple<Matrix, std::vector<float>> ShuffleInput(Matrix in, std::vector<float> labels) {

	// Shuffle input
	for (int k = 0; k < in.ColumnCount; k++) {

		int r = k + rand() % (in.ColumnCount - k);

		std::vector<float> tempI = in.Column(k);
		int tempL = labels[k];

		in.SetColumn(k, in.Column(r));
		labels[k] = labels[r];

		in.SetColumn(r, tempI);
		labels[r] = tempL;
	}

	return std::make_tuple(in, labels);
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
	inputLabels = std::vector<float>(imageNum);
	YTotal = Matrix(gpnn_dimensions[gpnn_dimensions.size() - 1], imageNum);

	for (int i = 0; i < imageNum; i++) {

		std::vector<uint8_t> byteData((width * height));
		trainingFR.read(reinterpret_cast<char*>(byteData.data()), byteData.size());
		std::vector<int> intData(byteData.begin(), byteData.end());

		input.SetColumn(i, intData);

		std::vector<int> y = std::vector<int>(gpnn_dimensions[gpnn_dimensions.size() - 1], 0);

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
	testLabels = std::vector<float>(imageNum);

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

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - sTime;
	std::cout << "Time to load input: " << (time.count() / 1000.00) << " seconds" << std::endl;

	input = input.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries, laguerreSeries, lowerNormalized, upperNormalized);
	testData = testData.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries, laguerreSeries, lowerNormalized, upperNormalized);

	gpnn_dimensions[0] = input.RowCount;
	vnn_dimensions[0] = input.RowCount;
}

int ReadBigInt(std::ifstream* fr) {

	int littleInt;
	fr->read(reinterpret_cast<char*>(&littleInt), sizeof(int));

	unsigned char* bytes = reinterpret_cast<unsigned char*>(&littleInt);
	std::swap(bytes[0], bytes[3]);
	std::swap(bytes[1], bytes[2]);

	return littleInt;
}


std::tuple<Matrix, std::vector<float>> MakeDataset(Matrix data, std::vector<float> labels, int num) {

	// Find all instances of num in data
	std::vector<int> target_index;
	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == num) {
			target_index.push_back(i);
		}
	}

	std::unordered_set<int> used (target_index.begin(), target_index.end());
	std::vector<int> value_index;

	// Get random instances that aren't num
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
	std::vector<float> dataset_labels;
	dataset_labels.reserve(all_indexes.size());

	// Actually make dataset
	for (int i = 0; i < all_indexes.size(); i++) {
		dataset.SetColumn(i, data.Column(all_indexes[i]));

		labels[all_indexes[i]] == num ? dataset_labels.push_back(1) : dataset_labels.push_back(0);
	}

	return std::make_tuple(dataset, dataset_labels);
}

 std::vector<float> MakeTestDataset(std::vector<float> labels, int num) {

	for (int i = 0; i < labels.size(); i++) {
		labels[i] == num ? labels[i] = 1 : labels[i] = 0;
	}

	return labels;
}


std::tuple<Matrix, std::vector<float>> GetNextInput(Matrix in, std::vector<float> labels, int size, int i) {
	std::vector<float> batch_labels = std::vector<float>();
	batch_labels.reserve(size);

	for (int x = i * size; x < i * size + size; x++) {
		batch_labels.push_back(labels[x]);
	}

	return std::make_tuple(in.SegmentC(i * size, i * size + size), batch_labels);
}


void TrainVNN() {

	std::chrono::steady_clock::time_point tStart;
	std::chrono::duration<double, std::milli> time;

	Matrix vnn_data;
	std::vector<float> vnn_labels;

	Matrix vnn_batch;
	std::vector<float> vnn_batch_labels;

	for (int v = 0; v < 10; v++) {

		std::cout << "Training Validation Network " << v  << ":" << std::endl;

		// Make dataset containing 50% target value
		std::tie(vnn_data, vnn_labels) = MakeDataset(input, inputLabels, v);
		int iterations = vnn_data.ColumnCount / batchSize;

		for (int e = 0; e < epochs; e++) {

			tStart = std::chrono::high_resolution_clock::now();

			std::tie(vnn_data, vnn_labels) = ShuffleInput(vnn_data, vnn_labels);

			for (int i = 0; i < iterations; i++) {

				std::tie(vnn_batch, vnn_batch_labels) = GetNextInput(vnn_data, vnn_labels, batchSize, i);

				std::tie(vnn_activation[v], vnn_total[v]) = ForwardPropogation(vnn_batch, vnn_weights[v], vnn_biases[v], vnn_activation[v], vnn_total[v], vnn_res);
				std::tie(vnn_weights[v], vnn_biases[v]) = BackwardPropogation(vnn_batch, vnn_batch_labels, vnn_weights[v], vnn_biases[v], vnn_activation[v], vnn_total[v], vnn_res);
			}

			std::vector<float> vnn_test_labels = MakeTestDataset(testLabels, v);

			float acc = TestNetwork(vnn_weights[v], vnn_biases[v], vnn_res, vnn_test_labels);

			time = std::chrono::high_resolution_clock::now() - tStart;
			std::cout << "Vnn: " << v << " Epoch: " << e << " Accuracy: " << acc << " Epoch Time: ";
			CleanTime(time.count());
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
		A[i] = i < Z.size() - 1 ? LeakyReLU(Z[i]) : Sigmoid(Z[i]);
	}

	return std::make_tuple( A, Z );
}

std::tuple<std::vector<Matrix>, std::vector<std::vector<float>> > BackwardPropogation(Matrix in, std::vector<float> labels, std::vector<Matrix> w, std::vector<std::vector<float>> b,
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
	dT[dT.size() - 1] = A[A.size() - 1] - labels;

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


float TestNetwork(std::vector<Matrix> w, std::vector<std::vector<float>> b, std::unordered_set<int> res, std::vector<float> labels) {

	Matrix current_total;
	Matrix last_activation;

	for (int i = 0; i < w.size(); i++) {
		current_total = Matrix(w[i].RowCount, testData.ColumnCount);
		if (res.find(i) != res.end()) {

			current_total.Insert(0, testData);
			last_activation.Insert(0, testData);

			current_total.Insert(testData.RowCount, (w[i].DotProduct(i == 0 ? testData : last_activation) + b[i]).Transpose());
		}
		else {
			current_total = (w[i].DotProduct(i == 0 ? testData : last_activation) + b[i]).Transpose();
		}
		last_activation = i < w.size() - 1 ? LeakyReLU(current_total) : Sigmoid(current_total);
	}

	// Calculate accuracy
	std::vector<float> predictions = std::vector<float>(last_activation.ColumnCount);
	int correct = 0;

	for (int i = 0; i < last_activation.ColumnCount; i++) {

		predictions[i] = last_activation.Column(i)[0];
		predictions[i] > 0.5f ? predictions[i] = 1 : predictions[i] = 0;

		if (predictions[i] == labels[i]) { correct++; }
	}
	return (float)correct / testData.ColumnCount;
}