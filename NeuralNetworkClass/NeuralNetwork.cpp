#include "NeuralNetwork.h"

void NeuralNetwork::Define(std::vector<int> dimensions, std::unordered_set<int> res_net, std::unordered_set<int> batch_normalization, 
	Matrix(Matrix::* activation_function)(), Matrix(Matrix::* activation_function_derivative)(), Matrix(Matrix::* end_activation_function)()) {

	this->res_net_layers = res_net;
	this->batch_norm_layers = batch_normalization;
	this->network_dimensions = dimensions;

	this->activation_function = activation_function;
	this->end_activation_function = end_activation_function;
	this->activation_function_derivative = activation_function_derivative;

	std::cout << "Network Summary {\ndims = ";
	for (int i = 0; i < network_dimensions.size() - 1; i++) {
		std::cout << network_dimensions[i] << "_";
	} std::cout << network_dimensions.back() << std::endl;

	std::cout << "}\n\n";
}

void NeuralNetwork::Compile(loss_metrics loss, loss_metrics metrics, optimization_technique optimizer, Matrix::init weight_initialization) {

	current_network.weights.reserve(network_dimensions.size() - 1);
	current_network.biases.reserve(network_dimensions.size() - 1);

	// Initialization of network matrices
	for (int i = 0; i < network_dimensions.size() - 1; i++) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			current_network.weights.emplace_back(network_dimensions[i] + network_dimensions[0], network_dimensions[i + 1], weight_initialization);
		} else {
			current_network.weights.emplace_back(network_dimensions[i], network_dimensions[i + 1], weight_initialization);
		}
		current_network.biases.emplace_back(network_dimensions[i + 1], 0);
	}

	this->loss = loss;
	switch (loss) {
	case loss_metrics::mse:
		loss_function = &NeuralNetwork::mse_loss;
		break;
	case loss_metrics::mae:
		loss_function = &NeuralNetwork::mae_loss;
		break;
	}

	std::cout << "Status: network_compiled\n";
}

NeuralNetwork::training_history NeuralNetwork::Fit(Matrix x_train, Matrix y_train, int batch_size, int epochs, float learning_rate, float validation_split, bool shuffle, int validation_freq) {

	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	result_matrices current_results;
	derivative_matrices current_derivs;

	training_history history;

	Matrix x_test;
	Matrix y_test;

	std::tie(current_results, current_derivs) = initialize_result_matrices(batch_size);

	std::tie(x_train, y_train, x_test, y_test) = data_preprocessing(x_train, y_train, shuffle, validation_split);

	const int iterations = x_train.ColumnCount / batch_size; 

	for (int e = 0; e < epochs; e++) {

		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations; i++) {

			Matrix x = x_train.SegmentC((i * batch_size), (batch_size + (i * batch_size)));
			Matrix y = y_train.SegmentC((i * batch_size), (batch_size + (i * batch_size)));

			current_results = forward_propogate(x, current_network, current_results);
			current_network = backward_propogate(x, y, learning_rate, current_network, current_results, current_derivs);
		}

		// Test network every n epochs
		std::string out;
		if (e % validation_freq == validation_freq - 1) {
			std::string score = test_network(x_test, y_test, current_network);

			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << " " << score << std::endl;
		} else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << std::endl;
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	history.train_time = end_time - start_time;
	history.epoch_time = (end_time - start_time) / epochs;

	std::cout << "Status: training_complete\n";

	return history;
}

Matrix NeuralNetwork::Predict(Matrix x_test) {
	result_matrices test_results;

	// Initialize test result matrices
	test_results.total.reserve(current_network.weights.size());
	test_results.activation.reserve(current_network.weights.size());

	for (int i = 0; i < current_network.weights.size(); i++) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			test_results.total.emplace_back(current_network.weights[i].ColumnCount + network_dimensions[0], x_test.ColumnCount);
		}
		else {
			test_results.total.emplace_back(current_network.weights[i].ColumnCount, x_test.ColumnCount);
		}
		test_results.activation.emplace_back(current_network.weights[i].ColumnCount, x_test.ColumnCount);
	}

	test_results = forward_propogate(x_test, current_network, test_results);

	return test_results.activation.back();
}


std::tuple<NeuralNetwork::result_matrices, NeuralNetwork::derivative_matrices> NeuralNetwork::initialize_result_matrices(int batch_size) {
	result_matrices results;
	derivative_matrices derivs;

	// Initialize result matrices and derivative matrices in local variables
	results.total.reserve(current_network.weights.size());
	results.activation.reserve(current_network.weights.size());
	derivs.d_weights.reserve(network_dimensions.size() - 1);
	derivs.d_biases.reserve(network_dimensions.size() - 1);

	for (int i = 0; i < current_network.weights.size(); i++) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			results.total.emplace_back(current_network.weights[i].ColumnCount + network_dimensions[0], batch_size);
		}
		else {
			results.total.emplace_back(current_network.weights[i].ColumnCount, batch_size);
		}
		results.activation.emplace_back(results.total[i].RowCount, batch_size);
		derivs.d_total.emplace_back(results.total[i].RowCount, batch_size);

		derivs.d_weights.emplace_back(current_network.weights[i].RowCount, current_network.weights[i].ColumnCount);
		derivs.d_biases.emplace_back(current_network.biases[i].size());
	}

	return std::make_tuple(results, derivs);
}

std::tuple<Matrix, Matrix, Matrix, Matrix> NeuralNetwork::data_preprocessing(Matrix x_train, Matrix y_train, bool shuffle, float validation_split) {
	Matrix x_test;
	Matrix y_test;

	if (shuffle) {
		std::tie(x_train, y_train) = Shuffle(x_train, y_train);
	}
	
	if (validation_split > 0.0f) {
		int elements = (float)x_test.ColumnCount * validation_split;
		x_test = x_train.SegmentC(x_train.ColumnCount - elements);
		y_test = y_train.SegmentC(x_train.ColumnCount - elements);

		x_train = x_train.SegmentC(0, elements);
		y_train = y_train.SegmentC(0, elements);
	}
	return std::make_tuple(x_train, y_train, x_test, y_test);
}


NeuralNetwork::result_matrices NeuralNetwork::forward_propogate(Matrix x, network_structure net, result_matrices results) {
	for (int i = 0; i < results.total.size(); i++) {
		if (res_net_layers.find(i) != res_net_layers.end()) {

			results.total[i].Insert(0, x);
			results.activation[i].Insert(0, x);

			results.total[i].Insert(x.RowCount, (net.weights[i].DotProduct(i == 0 ? x : results.activation[i - 1]) + net.biases[i]).Transpose());
		}
		else {
			results.total[i] = (net.weights[i].DotProduct(i == 0 ? x : results.activation[i - 1]) + net.biases[i]).Transpose();
		}
		results.activation[i] = i < results.total.size() - 1 ? (results.total[i].*activation_function)() : (results.total[i].*end_activation_function)();
	}

	return results;
}

NeuralNetwork::network_structure  NeuralNetwork::backward_propogate(Matrix x, Matrix y, float learning_rate, network_structure net, result_matrices results, derivative_matrices deriv) {

	// Compute loss
	deriv.d_total[deriv.d_total.size() - 1] = (this->*loss_function)(results.activation.back(), y);

	for (int i = deriv.d_total.size() - 2; i > -1; i--) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			deriv.d_total[i] = ((deriv.d_total[i + 1].DotProduct(net.weights[i + 1].SegmentR(x.RowCount))).Transpose() * (results.total[i].SegmentR(x.RowCount).*activation_function_derivative)());
		}
		else {
			deriv.d_total[i] = ((deriv.d_total[i + 1].DotProduct(net.weights[i + 1])).Transpose() * (results.total[i].*activation_function_derivative)());
		}
	}

	int i = 0;
	for (Matrix& d_weight : deriv.d_weights) {
		d_weight = (deriv.d_total[i].Transpose().DotProduct(i == 0 ? x.Transpose() : results.activation[i - 1].Transpose()) * (1.0f / (float)x.ColumnCount)).Transpose();
		deriv.d_biases[i] = deriv.d_total[i].Multiply(1.0f / (float)x.ColumnCount).RowSums();
		i++;
	}

	for (int i = 0; i < net.weights.size(); i++) {
		net.weights[i] -= deriv.d_weights[i].Multiply(learning_rate);
		for (int x = 0; x < net.biases[i].size(); x++) {
			net.biases[i][x] -= (deriv.d_biases[i][x] * learning_rate);
		}
	}

	return net;
}


std::string NeuralNetwork::test_network(Matrix x, Matrix y, network_structure net) {

	std::string out;

	result_matrices test_results;

	// Initialize test result matrices
	test_results.total.reserve(current_network.weights.size());
	test_results.activation.reserve(current_network.weights.size());

	for (int i = 0; i < current_network.weights.size(); i++) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			test_results.total.emplace_back(current_network.weights[i].ColumnCount + network_dimensions[0], x.ColumnCount);
		}
		else {
			test_results.total.emplace_back(current_network.weights[i].ColumnCount, x.ColumnCount);
		}
		test_results.activation.emplace_back(current_network.weights[i].ColumnCount, x.ColumnCount);
	}

	test_results = forward_propogate(x, net, test_results);

	switch (loss) {
	case loss_metrics::mse:
		out = "mse: ";
		break;
	case loss_metrics::mae:
		out = "mae: ";
		break;		
	}

	switch (loss) {
	case loss_metrics::mse:
	case loss_metrics::mae:
		Matrix error = (this->*loss_function)(test_results.activation.back(), y);
		float total_error = std::abs(error.RowSums()[0] / (float)error.ColumnCount);
		out = out.append(std::to_string(total_error));
		break;
	}

	return out;
}

std::tuple<Matrix, Matrix> NeuralNetwork::Shuffle(Matrix x, Matrix y) {
	for (int k = 0; k < x.ColumnCount; k++) {

		int r = k + rand() % (x.ColumnCount - k);

		std::vector<float> tempX = x.Column(k);
		std::vector<float> tempY = y.Column(k);

		x.SetColumn(k, x.Column(r));
		y.SetColumn(k, y.Column(r));

		x.SetColumn(r, tempX);
		y.SetColumn(r, tempY);
	}

	return std::make_tuple(x, y);
}


Matrix NeuralNetwork::mae_loss(Matrix final_activation, Matrix labels) {
	return (final_activation - labels);
}

Matrix NeuralNetwork::mse_loss(Matrix final_activation, Matrix labels) {
	return (final_activation - labels).Pow(2);
}


std::string NeuralNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append(" hours");
	}
	else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append(" minutes");
	}
	else if (time / second > 1.00) {
		out = std::to_string(time / second).append(" seconds");
	}
	else {
		out = std::to_string(time).append(" ms");
	}
	return out;
}