#include <iostream>

#include "NeuralNetwork.h"

int main()
{
	NeuralNetwork model = NeuralNetwork();

	std::vector<int> dims = { 784, 32, 10 };
	std::unordered_set<int> res = { 0 };
	std::unordered_set<int> batch_norm = { 0 };

	model.Define(dims, res, batch_norm);
}