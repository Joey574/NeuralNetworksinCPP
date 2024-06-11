#include "NeuralNetwork.h"

void NeuralNetwork::Define(std::vector<int> dimensions, std::unordered_set<int> res_net = {}, std::unordered_set<int> batch_normalization = {}) {

	res_net_layers = res_net;
	batch_norm_layers = batch_normalization;

	for (int i = 1; i < dimensions.size(); i++) {
		if (res_net.find(i) != res_net.end()) {
			network.weights.emplace_back(dimensions[i - 1] + dimensions[0], dimensions[i]);
		} else {
			network.weights.emplace_back(dimensions[i - 1], dimensions[i]);
		}
		network.biases.emplace_back(network.weights[i - 1].RowCount, 0);
	}
}

void NeuralNetwork::Fit(Matrix x_train, Matrix y_train, int batch_size, int epochs, float validation_split = 0.0f,
	bool shuffle = true, int validation_freq = 1) {

	const int iterations = x_train.ColumnCount / batch_size; 

	for (int e = 0; e < epochs; e++) {

		for (int i = 0; i < iterations; i++) {

		}
	}


}