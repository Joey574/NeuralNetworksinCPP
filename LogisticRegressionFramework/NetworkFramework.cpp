#include "NetworkFramework.h"

#include <iostream>


void NetworkFramework::SetNetworkDimensions(std::vector<int> dimensions) {
	networkDimensions = dimensions;
}

void NetworkFramework::SetInput(Matrix input, std::vector<int> labels, Matrix YTotal) {
	this->input = input;
	this->inputLabels = labels;
	this->YTotal = YTotal;
}

void NetworkFramework::SetTestData(Matrix testData, std::vector<int> testLabels) {
	this->testData = testData;
	this->testLabels = testLabels;
}

void NetworkFramework::SetActivationFunctions(Matrix(*activationFunction)(Matrix total, float alpha), Matrix(*derivativeFunction)(Matrix total, float alpha), float alpha) {
	this->activationFunction = activationFunction;
	this->derivativeFunction = derivativeFunction;
	this->ELUAlpha = alpha;
}

void NetworkFramework::SetHyperparameters(float learningRate, float thresholdAccuracy, int batchSize, int iterations) {
	this->learningRate = learningRate;
	this->thresholdAccuracy = thresholdAccuracy;
	this->batchSize = batchSize;
	this->iterations = iterations;
}


void NetworkFramework::InitializeNetwork() {
	auto initStart = std::chrono::high_resolution_clock::now();

	int connections = 0;

	weights = std::vector<Matrix>(networkDimensions.size() - 1);
	dWeights = std::vector<Matrix>(weights.size());

	biases = std::vector<std::vector<float>>(weights.size());
	dBiases = std::vector<std::vector<float>>(biases.size());

	for (int i = 0; i < weights.size(); i++) {
		weights[i] = Matrix(networkDimensions[i], networkDimensions[i + 1], -0.5f, 0.5f);
		connections += weights[i].ColumnCount * weights[i].RowCount;

		biases[i] = std::vector<float>(weights[i].Row(i));
		connections += biases[i].size();

		dWeights[i] = Matrix(weights[i].RowCount, weights[i].ColumnCount);
		dBiases[i] = std::vector<float>(biases[i].size());
	}

	FileSize = (((sizeof(float)) * (connections)) + ((sizeof(int) * networkDimensions.size() - 1))) / 1000000.00;

	InitializeResultMatrices(batchSize);

	auto initEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> initTime = initEnd - initStart;

	timeToInitialize = initTime.count();
}

void NetworkFramework::InitializeResultMatrices(int size) {
	aTotal = std::vector<Matrix>(weights.size());
	activation = std::vector<Matrix>(aTotal.size());
	dTotal = std::vector<Matrix>(aTotal.size());

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = Matrix(weights[i].ColumnCount, size);
		activation[i] = Matrix(aTotal[i].RowCount, size);
		dTotal[i] = Matrix(aTotal[i].RowCount, size);
	}
}

Matrix NetworkFramework::RandomizeInput(Matrix totalInput, int size) {
	Matrix a = Matrix(totalInput.RowCount, size);

	std::unordered_set<int> used = std::unordered_set<int>(size);

	YBatch = Matrix(networkDimensions[networkDimensions.size() - 1], size);
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


void NetworkFramework::TrainNetwork() {

	std::chrono::steady_clock::time_point totalStart = std::chrono::high_resolution_clock::now();
	std::chrono::steady_clock::time_point totalEnd;

	std::chrono::duration<double, std::milli> time;

	batch = RandomizeInput(input, batchSize);

	for (int i = 0; i < iterations; i++) {
		ForwardPropogation();
		BackwardPropogation();
		UpdateNetwork();

		float acc = Accuracy(GetPredictions(batchSize), batchLabels);

		if (acc > thresholdAccuracy) {
			batch = RandomizeInput(input, batchSize);
		}
	}

	totalEnd = std::chrono::high_resolution_clock::now();

	time = (totalEnd - totalStart);

	averageIterationTime = time.count() / iterations;
	totalTrainTime = (time / 1000.00).count();
}

float NetworkFramework::TestNetwork() {
	batch = testData;

	InitializeResultMatrices(testData.ColumnCount);
	ForwardPropogation();

	return Accuracy(GetPredictions(testData.ColumnCount), testLabels);
}


void NetworkFramework::ForwardPropogation() {

	for (int i = 0; i < aTotal.size(); i++) {
		aTotal[i] = (weights[i].DotProduct(i == 0 ? batch : activation[i - 1]) + biases[i]).Transpose();
		activation[i] = i < aTotal.size() - 1 ? (this->activationFunction)(aTotal[i], ELUAlpha) : SoftMax(aTotal[i]);
	}
}

void NetworkFramework::BackwardPropogation() {

	dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - YBatch;

	for (int i = dTotal.size() - 2; i > -1; i--) {
		dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])).Transpose() * (this->derivativeFunction)(aTotal[i], ELUAlpha));
	}

	std::for_each(std::execution::par_unseq, dWeights.begin(), dWeights.end(), [&](auto&& item) {
		size_t i = &item - dWeights.data();
		dWeights[i] = (dTotal[i].Transpose().DotProduct(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize)).Transpose();
		dBiases[i] = dTotal[i].Multiply(1.0f / (float)batchSize).RowSums();
	});
}

void NetworkFramework::UpdateNetwork() {
	for (int i = 0; i < weights.size(); i++) {
		weights[i] -= dWeights[i].Multiply(learningRate);
	}

	for (int i = 0; i < biases.size(); i++) {
		for (int x = 0; x < biases[i].size(); x++) {
			biases[i][x] -= (dBiases[i][x] * learningRate);
		}
	}
}


std::vector<int> NetworkFramework::GetPredictions(int len) {

	std::vector<int> predictions = std::vector<int>(len);

	for (int i = 0; i < len; i++) {

		std::vector<float> a = activation[activation.size() - 1].Column(i);

		auto maxElementIterator = std::max_element(a.begin(), a.end());
		predictions[i] = std::distance(a.begin(), maxElementIterator);
	}
	return predictions;
}

float NetworkFramework::Accuracy(std::vector<int> predictions, std::vector<int> labels) {
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



void NetworkFramework::SaveNetwork(std::string filename) {
	
}

void NetworkFramework::LoadNetwork(std::string filename) {
	
}