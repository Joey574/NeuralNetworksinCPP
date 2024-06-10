#include <iostream>

#include "NeuralNetwork.h"

int main()
{
	NeuralNetwork model = NeuralNetwork();

	std::vector<int> dims = { 784, 32, 10 };
	std::vector<int> res = { 0 };
	std::vector<int> batch_norm = { 0 };

	model.Define(dims, res, batch_norm);
}