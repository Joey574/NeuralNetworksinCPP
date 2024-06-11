#pragma once
#include <unordered_set>
#include <iostream>

#include "Matrix.h"

class NeuralNetwork
{
public:
	static enum class initialization_technique
	{
		random, normalized, xavier, he
	};

	static enum class loss_metrics {
		none, mse, mae, accuracy
	};

	static enum class optimization_technique {
		none
	};

	void Define(std::vector<int> dimensions, std::unordered_set<int> res_net = {}, std::unordered_set<int> batch_normalization = {});

	void Compile(loss_metrics loss = loss_metrics::none, loss_metrics metrics = loss_metrics::none, 
		optimization_technique optimizer = optimization_technique::none, 
		initialization_technique weight_initialization = initialization_technique::random);

	void Fit(Matrix x_train, Matrix y_train, int batch_size, int epochs, float validation_split = 0.0f, 
		bool shuffle = true, int validation_freq = 1);

	std::tuple<Matrix, Matrix> Shuffle(Matrix x, Matrix y);

	std::string Evaluate(Matrix x_train, Matrix y_train);

	Matrix Predict(Matrix x_test);

private:

	struct network_structure {
		std::vector<Matrix> weights;
		std::vector<std::vector<float>> biases;
	};

	struct result_matrices {
		std::vector<Matrix> total;
		std::vector<Matrix> activation;
	};

	network_structure network;

	std::unordered_set<int> res_net_layers;
	std::unordered_set<int> batch_norm_layers;

	result_matrices ForwardPropogation(network_structure network, Matrix x);
	network_structure BackwardPropogation(network_structure network, result_matrices results);
	std::string TestNetwork(network_structure network);

	std::string clean_time(double time);
};

