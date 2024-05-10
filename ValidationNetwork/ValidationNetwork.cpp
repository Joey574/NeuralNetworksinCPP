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
#include "VnnHelpers.h"
#include "DnnHelpers.h"

// Hyperparameters
std::vector<int> vnn_dimensions = { 784, 30, 30, 1 };
std::vector<int> gpnn_dimensions = { 784, 32, 32, 10 };
std::vector<int> dnn_dimensions = { 20, 32, 32, 32, 10 };

std::unordered_set<int> vnn_res = {  }; 
std::unordered_set<int> gpnn_res = {  }; 
std::unordered_set<int> dnn_res = {  }; 

float lowerNormalized = 0.0f;
float upperNormalized = 1.0f;
Matrix::init initType = Matrix::init::He;
int batchSize = 500;
float learningRate = 0.02;

int vnn_epochs = 1;
int dnn_epochs = 10;

int test_a_size = 1000;

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

// VNN weights, biases, and results
std::vector<std::vector<Matrix>> vnn_weights;
std::vector<std::vector<std::vector<float>>> vnn_biases;

std::vector<std::vector<Matrix>> vnn_total;
std::vector<std::vector<Matrix>> vnn_activation;

std::vector<Matrix> vnn_final_activations;
std::vector<float> vnn_final_accuracy;

// DNNweights, biases, and results
std::vector<Matrix> dnn_weights;
std::vector<std::vector<float>> dnn_biases;

std::vector<Matrix> dnn_total;
std::vector<Matrix> dnn_activation;


// Prototypes
void CleanTime(double time);
void InitializeNetworks();
void LoadInput();
int ReadBigInt(std::ifstream* fr);

std::tuple<Matrix, std::vector<float>> MakeDataset(Matrix data, std::vector<float> labels, int num);
std::vector<float> MakeTestDataset(std::vector<float> labels, int num);
std::tuple<Matrix, std::vector<float>> GetNextInput(Matrix in, std::vector<float> labels, int size, int i);
std::tuple<Matrix, Matrix, std::vector<float>> GetNextInput(Matrix in, Matrix y, std::vector<float> labels, int size, int i);
void TrainVNN(int epochs, Matrix vnn_test_data, std::vector<float> vnn_test_data_labels);
void TrainGPNN();
void TrainDNN(int epochs, Matrix dnn_test_data, std::vector<float> dnn_test_data_labels);
std::tuple<std::vector<Matrix>, std::vector<Matrix>> ForwardPropogation(Matrix in, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res, Matrix(*operation)(Matrix mat));
Matrix LastActivation(std::vector<Matrix> w, std::vector<std::vector<float>> b, std::unordered_set<int> res, Matrix test_data, Matrix(*operation)(Matrix mat));


int main()
{
	LoadInput();

	InitializeNetworks();

	// Get first 1000 data points in test set A
	std::vector<float> test_a_labels;
	for (int i = 0; i < test_a_size; i++) {
		test_a_labels.push_back(testLabels[i]);
	}

	std::vector<float> test_b_labels;
	for (int i = test_a_size; i < testLabels.size(); i++) {
		test_b_labels.push_back(testLabels[i]);
	}

	TrainVNN(vnn_epochs, testData.SegmentC(0, test_a_size), test_a_labels);

	TrainDNN(dnn_epochs, testData.SegmentC(test_a_size), test_b_labels);
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

	// Initialize vnn
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

	// TODO: Initialize gpnn

	// Initialize dnn
	dnn_weights.reserve(dnn_dimensions.size() - 1);
	dnn_biases.reserve(dnn_dimensions.size() - 1);

	dnn_total.reserve(dnn_dimensions.size() - 1);
	dnn_activation.reserve(dnn_dimensions.size() - 1);

	for (int i = 0; i < dnn_dimensions.size() - 1; i++) {

		if (dnn_res.find(i - 1) != dnn_res.end()) {
			dnn_weights.emplace_back(dnn_dimensions[i] + dnn_dimensions[0], dnn_dimensions[i + 1], initType);
		}
		else {
			dnn_weights.emplace_back(dnn_dimensions[i], dnn_dimensions[i + 1], initType);
		}
		dnn_biases.emplace_back(std::vector<float>(dnn_dimensions[i + 1], 0));
	}

	for (int i = 0; i < dnn_weights.size(); i++) {
		if (dnn_res.find(i) != dnn_res.end()) {
			dnn_total.emplace_back(dnn_weights[i].ColumnCount + dnn_dimensions[0], batchSize);
		}
		else {
			dnn_total.emplace_back(dnn_weights[i].ColumnCount, batchSize);
		}
		dnn_activation.emplace_back(dnn_weights[i].RowCount, batchSize);
	}
	

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
	std::cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << std::endl;
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

std::tuple<Matrix, Matrix, std::vector<float>> GetNextInput(Matrix in, Matrix y, std::vector<float> labels, int size, int i) {
	std::vector<float> batch_labels = std::vector<float>();
	batch_labels.reserve(size);

	for (int x = i * size; x < i * size + size; x++) {
		batch_labels.push_back(labels[x]);
	}

	return std::make_tuple(in.SegmentC(i * size, i * size + size), y.SegmentC(i * size, i * size + size), batch_labels);
}


void TrainVNN(int epochs, Matrix vnn_test_data, std::vector<float> vnn_test_data_labels) {

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
		float acc;

		for (int e = 0; e < epochs; e++) {

			tStart = std::chrono::high_resolution_clock::now();
			std::tie(vnn_data, vnn_labels) = ShuffleInput(vnn_data, vnn_labels);

			for (int i = 0; i < iterations; i++) {

				std::tie(vnn_batch, vnn_batch_labels) = GetNextInput(vnn_data, vnn_labels, batchSize, i);

				std::tie(vnn_activation[v], vnn_total[v]) = ForwardPropogation(vnn_batch, vnn_weights[v], vnn_biases[v], vnn_activation[v], vnn_total[v], vnn_res, &Sigmoid);
				std::tie(vnn_weights[v], vnn_biases[v]) = BackwardPropogation(vnn_batch, vnn_batch_labels, vnn_weights[v], vnn_biases[v], vnn_activation[v], vnn_total[v], vnn_res, learningRate);
			}

			std::vector<float> vnn_test_labels = MakeTestDataset(vnn_test_data_labels, v);

			acc = VNN_Accuracy(LastActivation(vnn_weights[v], vnn_biases[v], vnn_res, vnn_test_data, &Sigmoid), vnn_test_labels);

			time = std::chrono::high_resolution_clock::now() - tStart;
			std::cout << "Vnn: " << v << " Epoch: " << e << " Accuracy: " << acc << " Epoch Time: ";
			CleanTime(time.count());
		}

		// Get and store activation of all training samples
		std::tie(vnn_activation[v], vnn_total[v]) = ForwardPropogation(input, vnn_weights[v], vnn_biases[v], vnn_activation[v], vnn_total[v], vnn_res, &Sigmoid);
		vnn_final_activations.push_back(vnn_activation[v][vnn_activation[v].size() - 1]);

		// Most recent accuracy score on first test dataset
		vnn_final_accuracy.push_back(acc);

		// TODO: Rollback to best weights after network training done
	}
}

void TrainGPNN() {

}

void TrainDNN(int epochs, Matrix test_b_data, std::vector<float> dnn_test_data_labels) {

	Matrix dnn_dataset = Matrix(0, input.ColumnCount);
	Matrix dnn_y = YTotal;
	std::vector<float> dnn_dataset_labels = inputLabels;

	Matrix dnn_batch_dataset;
	Matrix dnn_batch_y;
	std::vector<float> dnn_batch_labels;

	Matrix dnn_test_data = Matrix(0, test_b_data.ColumnCount);

	// Make dataset
	for (int v = 0; v < vnn_final_activations.size(); v++) {
		dnn_dataset = dnn_dataset.Combine(vnn_final_activations[v]);
	}

	// Make test dataset
	for (int v = 0; v < vnn_final_activations.size(); v++) {
		dnn_test_data = dnn_test_data.Combine(LastActivation(vnn_weights[v], vnn_biases[v], vnn_res, test_b_data, &Sigmoid));
	}

	// Add accuracy on test_a to dataset
	for (int v = 0; v < vnn_final_accuracy.size(); v++) {
		std::vector<float> temp = std::vector<float>(input.ColumnCount, vnn_final_accuracy[v]);
		Matrix a = Matrix(1, temp.size());
		a.SetRow(0, temp);
		
		dnn_dataset = dnn_dataset.Combine(a);
	}

	// Add accuracy on test_a to test_b dataset
	for (int v = 0; v < vnn_final_accuracy.size(); v++) {
		std::vector<float> temp = std::vector<float>(test_b_data.ColumnCount, vnn_final_accuracy[v]);
		Matrix a = Matrix(1, temp.size());
		a.SetRow(0, temp);

		dnn_test_data = dnn_test_data.Combine(a);
	}

	std::chrono::steady_clock::time_point tStart;
	std::chrono::duration<double, std::milli> time;
	std::cout << "Training Decision Network:" << std::endl;

	int iterations = dnn_dataset.ColumnCount / batchSize;

	for (int e = 0; e < epochs; e++) {

		tStart = std::chrono::high_resolution_clock::now();
		std::tie(dnn_batch_dataset, dnn_batch_y, dnn_batch_labels) = ShuffleInput(dnn_dataset, dnn_y, dnn_dataset_labels);

		for (int i = 0; i < iterations; i++) {
			std::tie(dnn_batch_dataset, dnn_batch_y, dnn_batch_labels) = GetNextInput(dnn_dataset, dnn_y, dnn_dataset_labels, batchSize, i);

			std::tie(dnn_activation, dnn_total) = ForwardPropogation(dnn_batch_dataset, dnn_weights, dnn_biases, dnn_activation, dnn_total, dnn_res, &SoftMax);
			std::tie(dnn_weights, dnn_biases) = BackwardPropogation(dnn_batch_dataset, dnn_batch_y, dnn_weights, dnn_biases, dnn_activation, dnn_total, dnn_res, learningRate);
		}

		float acc = DNN_Accuracy(LastActivation(dnn_weights, dnn_biases, dnn_res, dnn_test_data, &SoftMax), dnn_test_data_labels);

		time = std::chrono::high_resolution_clock::now() - tStart;
		std::cout << "Dnn Epoch: " << e << " Accuracy: " << acc << " Epoch Time: ";
		CleanTime(time.count());
	}
}


std::tuple<std::vector<Matrix>, std::vector<Matrix>> ForwardPropogation(Matrix in, std::vector<Matrix> w, std::vector<std::vector<float>> b, 
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res, Matrix (*operation)(Matrix mat)) {

	for (int i = 0; i < Z.size(); i++) {
		if (res.find(i) != res.end()) {

			Z[i].Insert(0, in);
			A[i].Insert(0, in);

			Z[i].Insert(in.RowCount, (w[i].DotProduct(i == 0 ? in : A[i - 1]) + b[i]).Transpose());
		}
		else {
			Z[i] = (w[i].DotProduct(i == 0 ? in : A[i - 1]) + b[i]).Transpose();
		}
		A[i] = i < Z.size() - 1 ? LeakyReLU(Z[i]) : (*operation)(Z[i]);
	}

	return std::make_tuple( A, Z );
}


Matrix LastActivation(std::vector<Matrix> w, std::vector<std::vector<float>> b, std::unordered_set<int> res, Matrix test_data, Matrix(*operation)(Matrix mat)) {
	
	Matrix current_total;
	Matrix last_activation;

	for (int i = 0; i < w.size(); i++) {
		current_total = Matrix(w[i].RowCount, test_data.ColumnCount);
		if (res.find(i) != res.end()) {

			current_total.Insert(0, test_data);
			last_activation.Insert(0, test_data);

			current_total.Insert(test_data.RowCount, (w[i].DotProduct(i == 0 ? test_data : last_activation) + b[i]).Transpose());
		}
		else {
			current_total = (w[i].DotProduct(i == 0 ? test_data : last_activation) + b[i]).Transpose();
		}
		last_activation = i < w.size() - 1 ? LeakyReLU(current_total) : (*operation)(current_total);
	}
	return last_activation;
}