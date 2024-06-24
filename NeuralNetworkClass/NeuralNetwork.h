#pragma once
#include <unordered_set>
#include <iostream>

#include "Matrix.h"

class NeuralNetwork
{
public:

	static enum class loss_metrics {
		none, mse, mae
	};

	static enum class optimization_technique {
		none
	};

	void Define(
		std::vector<int> dimensions, 
		std::unordered_set<int> res_net, 
		std::unordered_set<int> batch_normalization, 
		Matrix(Matrix::* activation_function)(), 
		Matrix(Matrix::* activation_function_derivative)(),
		Matrix(Matrix::* end_activation_function)()
	);

	void Compile(
		loss_metrics loss = loss_metrics::none,
		loss_metrics metrics = loss_metrics::none,
		optimization_technique optimizer = optimization_technique::none,
		Matrix::init weight_initialization = Matrix::init::Random
	);

	void Fit(
		Matrix x_train,
		Matrix y_train,
		int batch_size,
		int epochs,
		float validation_split = 0.0f,
		bool shuffle = true,
		int validation_freq = 1
	);

	std::tuple<Matrix, Matrix> Shuffle(Matrix x, Matrix y);

	std::string Evaluate(Matrix x_train, Matrix y_train);

	Matrix Predict(Matrix x_test);

private:

	// Network Structs
	struct network_structure {
		std::vector<Matrix> weights;
		std::vector<std::vector<float>> biases;
	};

	struct result_matrices {
		std::vector<Matrix> total;
		std::vector<Matrix> activation;
	};

	struct derivative_matrices {
		std::vector<Matrix> d_total;
		std::vector<Matrix> d_weights;
		std::vector<std::vector<float>> d_biases;
	};

	// Function Pointers
	Matrix (Matrix::* activation_function)();
	Matrix (Matrix::* end_activation_function)();
	Matrix (Matrix::* activation_function_derivative)();

	Matrix(NeuralNetwork::* loss_function)(Matrix final_activation, Matrix labels);

	// Network
	network_structure current_network;
	result_matrices current_results;
	derivative_matrices current_derivs;

	std::vector<int> network_dimensions;

	std::unordered_set<int> res_net_layers;
	std::unordered_set<int> batch_norm_layers;

	result_matrices forward_propogate(Matrix x, network_structure net, result_matrices results);
	network_structure backward_propogate(Matrix x, Matrix y, network_structure net, result_matrices results, derivative_matrices deriv);
	std::string test_network(Matrix x, Matrix y, network_structure net);

	Matrix mse_loss(Matrix final_activation, Matrix labels);
	Matrix mae_loss(Matrix final_activation, Matrix labels);

	std::string clean_time(double time);
};