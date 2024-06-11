#include <iostream>

#include "NeuralNetwork.h"

int main()
{
	NeuralNetwork model = NeuralNetwork();

	std::vector<int> dims = { 784, 32, 10 };
	std::unordered_set<int> res = { 0 };
	std::unordered_set<int> batch_norm = { 0 };

	NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::mse;
	NeuralNetwork::loss_metrics eval_metric = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	NeuralNetwork::initialization_technique init_tech = NeuralNetwork::initialization_technique::he;

		
	Matrix x;
	Matrix y;

	int batch_size = 264;
	int epochs = 10;
	float valid_split = 0.0f;
	int valid_freq = 5;

	model.Define(dims, res, batch_norm);

	model.Compile(loss, eval_metric, optimizer, init_tech);

	model.Fit(x, y, batch_size, epochs, valid_split, true, valid_freq);
}