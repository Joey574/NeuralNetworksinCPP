#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>
#include <thread>
#include <iomanip>
#include <cmath>
#include <tuple>

#include "Matrix.h"
#include "ActivationFunctions.h"

std::tuple<std::vector<Matrix>, std::vector<std::vector<float>> > BackwardPropogation(Matrix in, std::vector<float> labels, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res, float learning_rate);
float VNN_Accuracy(Matrix last_activation, std::vector<float> labels);
std::tuple<Matrix, std::vector<float>> ShuffleInput(Matrix in, std::vector<float> labels);
