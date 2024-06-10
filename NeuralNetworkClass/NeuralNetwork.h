#pragma once
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
		none, threshold_stepdown
	};

	void Define(std::vector<int> dimensions, std::vector<int> res_net = std::vector<int>(0), std::vector<int> batch_normalization = std::vector<int>(0));

	void Compile(loss_metrics loss = loss_metrics::none, loss_metrics metrics = loss_metrics::none, optimization_technique optimizer = optimization_technique::none, initialization_technique weight_initialization = initialization_technique::random);

	void Fit(Matrix x_train, Matrix y_train, int batch_size, int epochs, float validation_split = 0.0f, bool shuffle = true, int validation_freq = 1);

	std::string Evaluate(Matrix x_train, Matrix y_train);

	Matrix Predict(Matrix x_test);

private:

	struct network_structure {
		std::vector<Matrix> weights;
		std::vector<std::vector<float>> biases;
	};

	network_structure ForwardPropogation(network_structure network);
	network_structure BackwardPropogation(network_structure network);
	
};

