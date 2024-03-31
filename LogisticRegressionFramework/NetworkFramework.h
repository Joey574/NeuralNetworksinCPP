#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <unordered_set>

#include "Matrix.h"
#include "ActivationFunctions.h"

class NetworkFramework
{
public:

	void SetNetworkDimensions(std::vector<int> dimensions);
	void SetInput(Matrix input, std::vector<int> labels, Matrix YTotal);
	void SetTestData(Matrix input, std::vector<int> labels);
	void SetHyperparameters(float learningRate,	float thresholdAccuracy, int batchSize,	int iterations);

	void SetActivationFunctions(Matrix(*activationFunction)(Matrix total, float alpha), Matrix(*derivativeFunction)(Matrix total, float alpha),  float alpha = 1.0f);

	void InitializeNetwork();

	void TrainNetwork();
	float TestNetwork();

	void SaveNetwork(std::string filename);
	void LoadNetwork(std::string filename);

	// Time values
	double timeToInitialize;
	double totalTrainTime;
	float averageIterationTime;

	// Misc values
	float FileSize;

private:

	void InitializeResultMatrices(int size);
	void ForwardPropogation();
	void BackwardPropogation();
	void UpdateNetwork();

	Matrix RandomizeInput(Matrix totalInput, int size);

	std::vector<int> GetPredictions(int len);
	float Accuracy(std::vector<int> predictions, std::vector<int> labels);

	// Hyperparameters
	std::vector<int> networkDimensions;
	float learningRate;
	float thresholdAccuracy;
	int batchSize;
	int iterations;

	float ELUAlpha;

	// Inputs
	Matrix input;
	Matrix batch;
	std::vector<int> inputLabels;
	std::vector<int> batchLabels;

	// Test Data
	Matrix testData;
	std::vector<int> testLabels;

	// Neural Network Matrices
	std::vector<Matrix> weights;
	std::vector<std::vector<float>> biases;

	// Result Matrices
	std::vector<Matrix> activation;
	std::vector<Matrix> aTotal;

	// Derivative Matrices
	std::vector<Matrix> dTotal;
	std::vector<Matrix> dWeights;
	std::vector<std::vector<float>> dBiases;

	// Error stuff
	Matrix YTotal;
	Matrix YBatch;

	// Pointers
	Matrix (*activationFunction)(Matrix total, float alpha);
	Matrix (*derivativeFunction)(Matrix total, float alpha);
};

