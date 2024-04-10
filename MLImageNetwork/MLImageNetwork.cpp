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
#include <bitset>
#include <atlimage.h>

#include "ActivationFunctions.h"
#include "Matrix.h"


// Hyperparameters
std::vector<int> dimensions = { 2, 16, 16, 1 };
std::unordered_set<int> resNet = {  };
int fourierSeries = 128;

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

Matrix::init initType = Matrix::init::He;
int epochs = 250;
int batchSize = 256;
float learningRate = 0.1;

// Image drawing stuff
Matrix unshuffledInput;
std::vector<int> unshuffledLabels;

// Inputs
Matrix input;
Matrix batch;

std::vector<int> inputLabels;
std::vector<float> batchLabels;

// Neural Network Matrices
std::vector<Matrix> weights;
std::vector<std::vector<float>> biases;

std::vector<Matrix> activation;
std::vector<Matrix> aTotal;

std::vector<Matrix> dTotal;
std::vector<Matrix> dWeights;
std::vector<std::vector<float>> dBiases;

// Prototypes
CImage LoadBMP(std::string filename);
void MakeBMP(std::string filename, std::vector<int> pixelData, CImage image);
void InitializeNetwork();
void InitializeResultMatrices(int size);
void TrainNetwork(CImage image);
void ForwardPropogation(Matrix in);
void BackwardPropogation();
void UpdateNetwork();
Matrix GetNextInput(Matrix totalInput, int size, int i);
std::vector<int> GetPredictions(int len);
float Accuracy(std::vector<int> predictions, std::vector<int> labels);
void CleanTime(double time);
std::wstring NarrowToWide(const std::string& narrowStr);


int main()
{
	srand(time(0));

	CImage t = LoadBMP("ML Images\\HelloWorld.bmp");

	InitializeNetwork();
	
	TrainNetwork(t);
}

void ShuffleInput() {
	for (int k = 0; k < input.ColumnCount; k++) {

		int r = k + rand() % (input.ColumnCount - k);

		std::vector<float> tempI = input.Column(k);
		int tempL = inputLabels[k];

		input.SetColumn(k, input.Column(r));
		inputLabels[k] = inputLabels[r];

		input.SetColumn(r, tempI);
		inputLabels[r] = tempL;
	}
}

std::wstring NarrowToWide(const std::string& narrowStr) {
	int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, nullptr, 0);
	std::wstring wideStr(wideStrLength, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, &wideStr[0], wideStrLength);
	return wideStr;
}

CImage LoadBMP(std::string filename) {
	auto sTime = std::chrono::high_resolution_clock::now();

	std::wstring fp = NarrowToWide(filename);

	CImage image;

	if (image.Load(fp.c_str()) == S_OK) {
		std::cout << "Image Loaded" << std::endl;
	}
	else {
		std::cout << "Image not found..." << std::endl;
	}

	input = Matrix(2, image.GetHeight() * image.GetWidth());

	for (int y = 0; y < image.GetHeight(); y++) {
		for (int x = 0; x < image.GetWidth(); x++) {
			inputLabels.push_back((image.GetPixel(x, y) / 16777215));
			input.SetColumn(x + (y * image.GetWidth()), std::vector<int>{x, y});
		}
	}

	// Normalize Input
	input = input.Normalized(lowerNormalized, upperNormalized);

	std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - sTime;
	std::cout << "Time to load image: " << (time.count() / 1000.00) << " seconds" << std::endl;

	Matrix oldI = input;
	// Compute Fourier Series
	if (fourierSeries > 0) {
		sTime = std::chrono::high_resolution_clock::now();
		std::cout << "Computing " << fourierSeries << " order(s) of Fourier Series..." << std::endl;

		for (int f = 0; f < fourierSeries; f++) {
			input = input.Combine(oldI.FourierSeries(f + 1));
		}
		dimensions[0] = input.RowCount;

		std::cout << "Fourier Features: " << input.RowCount - oldI.RowCount << std::endl;

		time = std::chrono::high_resolution_clock::now() - sTime;
		std::cout << "Time to compute " << fourierSeries << " order(s): " << time.count() / 1000.00 << " seconds" << std::endl;
	}

	// Save input data for image drawing
	unshuffledInput = input;
	unshuffledLabels = inputLabels;

	return image;
}

void MakeBMP(std::string filename, std::vector<int> pixelData, CImage image) {

	std::wstring fp = NarrowToWide(filename);

	CImage save;
	save.Create(image.GetWidth(), image.GetHeight(), image.GetBPP());

	RGBQUAD colors[2] = { 0 };
	save.GetColorTable(0, 2, colors);
	colors[1].rgbRed = colors[1].rgbGreen = colors[1].rgbBlue = 0xff;
	save.SetColorTable(0, 2, colors);

	for (int y = 0; y < save.GetHeight(); y++) {
		for (int x = 0; x < save.GetWidth(); x++) {
			save.SetPixel(x, y, pixelData[(y * save.GetWidth()) + x] == 0 ? RGB(0, 0, 0) : RGB(255,255,255));
		}
	}

	save.Save(fp.c_str(), Gdiplus::ImageFormatBMP);
}

Matrix GetNextInput(Matrix totalInput, int size, int i) {
	Matrix a = Matrix(totalInput.RowCount, size);

	a = totalInput.SegmentC(i * size, i * size + size);
	batchLabels.clear();

	for (int x = i * size; x < i * size + size; x++) {
		batchLabels.push_back(inputLabels[x]);
	}

	return a;
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

void TrainNetwork(CImage image) {
	std::cout << "TRAINING STARTED" << std::endl;

	std::chrono::steady_clock::time_point totalStart;
	std::chrono::steady_clock::time_point tStart;
	std::chrono::duration<double, std::milli> time;
	std::chrono::duration<double, std::milli> timeToReachHighest;

	std::vector<int> bestPredictions;

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

		InitializeResultMatrices(unshuffledInput.ColumnCount);
		ForwardPropogation(unshuffledInput);

		std::vector<int> predic = GetPredictions(unshuffledInput.ColumnCount);
		float acc = Accuracy(predic, unshuffledLabels);

		std::string filename = "NetworkImages\\" + std::to_string(e).append("_").append(std::to_string(acc)).append(".bmp");
		MakeBMP(filename, predic, image);

		if (acc >= highestAcc) {

			bestPredictions = GetPredictions(unshuffledInput.ColumnCount);
			highestAcc = acc;
			highestIndex = e;
			timeToReachHighest = std::chrono::high_resolution_clock::now() - totalStart;
		}

		InitializeResultMatrices(batchSize);

		time = std::chrono::high_resolution_clock::now() - tStart;
		std::cout << "Epoch: " << e << " Accuracy: " << acc << " Epoch Time: ";
		CleanTime(time.count());
	}

	MakeBMP("NetworkImages\\final.bmp",bestPredictions, image);

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
		activation[i] = i < aTotal.size() - 1 ? LeakyReLU(aTotal[i]) : Sigmoid(aTotal[i]);
	}
}

void BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - batchLabels;

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

std::vector<int> GetPredictions(int len) {

	std::vector<int> predictions = std::vector<int>(len);

	for (int i = 0; i < len; i++) {
		if (activation[activation.size() - 1].Column(i)[0] > 0.5f) {
			predictions[i] = 1;
		}
		else {
			predictions[i] = 0;
		}
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