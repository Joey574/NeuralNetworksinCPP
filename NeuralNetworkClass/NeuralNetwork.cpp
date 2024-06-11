#include "NeuralNetwork.h"

void NeuralNetwork::Define(std::vector<int> dimensions, std::unordered_set<int> res_net, std::unordered_set<int> batch_normalization) {

	res_net_layers = res_net;
	batch_norm_layers = batch_normalization;

	for (int i = 1; i < dimensions.size(); i++) {
		if (res_net.find(i) != res_net.end()) {
			network.weights.emplace_back(dimensions[i - 1] + dimensions[0], dimensions[i]);
		} else {
			network.weights.emplace_back(dimensions[i - 1], dimensions[i]);
		}
		network.biases.emplace_back(dimensions[i], 0);
	}
}

void NeuralNetwork::Compile(loss_metrics loss, loss_metrics metrics, optimization_technique optimizer, initialization_technique weight_initialization) {

}

void NeuralNetwork::Fit(Matrix x_train, Matrix y_train, int batch_size, int epochs, float validation_split,	bool shuffle, int validation_freq) {
	
	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	if (shuffle) {
		std::tie(x_train, y_train) = Shuffle(x_train, y_train);
	}

	const int iterations = x_train.ColumnCount / batch_size; 

	for (int e = 0; e < epochs; e++) {

		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < iterations; i++) {

			result_matrices results;

			Matrix x = x_train.SegmentC((i * batch_size), (batch_size + (i * batch_size)));

			results = ForwardPropogation(network, x);
			network = BackwardPropogation(network, results, x);
		}

		if (e % validation_freq == validation_freq - 1) {
			std::string score = TestNetwork(network);

			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << score << std::endl;
		} else {
			time = std::chrono::high_resolution_clock::now() - epoch_start_time;
			std::cout << "Epoch: " << e << " Time: " << clean_time(time.count()) << std::endl;
		}

		
	}
}

NeuralNetwork::result_matrices NeuralNetwork::ForwardPropogation(network_structure net, Matrix x) {

	result_matrices results;

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

NeuralNetwork::network_structure  NeuralNetwork::BackwardPropogation(network_structure net, result_matrices results, Matrix x) {

	derivative_matrices deriv;

	// do loss stuff
	deriv.d_total[deriv.d_total.size() - 1];

	for (int i = deriv.d_total.size() - 2; i > -1; i--) {
		if (res_net_layers.find(i) != res_net_layers.end()) {
			deriv.d_total[i] = ((deriv.d_total[i + 1].DotProduct(net.weights[i + 1].SegmentR(x.RowCount))).Transpose() * (results.total[i].SegmentR(x.RowCount).*activation_function_derivative)());
		}
		else {
			deriv.d_total[i] = ((deriv.d_total[i + 1].DotProduct(net.weights[i + 1])).Transpose() * results.total[i].ELUDerivative());
		}
	}

	std::for_each(std::execution::par_unseq, deriv.d_weights.begin(), deriv.d_weights.end(), [&](auto&& item) {
		size_t i = &item - deriv.d_weights.data();
		item = (deriv.d_total[i].Transpose().DotProduct(i == 0 ? x.Transpose() : results.activation[i - 1].Transpose()) * (1.0f / (float)x.ColumnCount)).Transpose();
		dBiases[i] = deriv.d_total[i].Multiply(1.0f / (float)x.ColumnCount).RowSums();
		});

	return net;
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

std::string NeuralNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append("hours");
	}
	else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append("minutes");
	}
	else if (time / second > 1.00) {
		out = std::to_string(time / second).append("seconds");
	}
	else {
		out = std::to_string(time).append("ms");
	}
	return out;
}