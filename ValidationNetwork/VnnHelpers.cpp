#include "VnnHelpers.h"


std::tuple<std::vector<Matrix>, std::vector<std::vector<float>> > BackwardPropogation(Matrix in, std::vector<float> labels, std::vector<Matrix> w, std::vector<std::vector<float>> b,
	std::vector<Matrix> A, std::vector<Matrix> Z, std::unordered_set<int> res, float learning_rate) {

	std::vector<Matrix> dT = std::vector<Matrix>();
	std::vector<Matrix> dW = std::vector<Matrix>();
	std::vector<std::vector<float>> dB = std::vector<std::vector<float>>();

	for (int i = 0; i < w.size(); i++) {
		dT.emplace_back(A[i].RowCount, A[i].ColumnCount);

		dW.emplace_back(w[i].RowCount, w[i].ColumnCount);
		dB.emplace_back(b[i].size());
	}

	// Backward prop
	dT[dT.size() - 1] = A[A.size() - 1] - labels;

	for (int i = dT.size() - 2; i > -1; i--) {

		if (res.find(i) != res.end()) {
			dT[i] = ((dT[i + 1].DotProduct(w[i + 1].SegmentR(in.RowCount))).Transpose() * LeakyReLUDerivative(Z[i].SegmentR(in.RowCount)));
		}
		else {
			dT[i] = ((dT[i + 1].DotProduct(w[i + 1])).Transpose() * LeakyReLUDerivative(Z[i]));
		}
	}

	std::for_each(std::execution::par, dW.begin(), dW.end(), [&](auto&& item) {
		size_t i = &item - dW.data();
		item = (dT[i].Transpose().DotProduct(i == 0 ? in.Transpose() : A[i - 1].Transpose()) * (1.0f / (float)in.ColumnCount)).Transpose();
		dB[i] = dT[i].Multiply(1.0f / (float)in.ColumnCount).RowSums();
		});


	// Update Network
	for (int i = 0; i < w.size(); i++) {
		w[i] -= dW[i].Multiply(learning_rate);
	}


	for (int i = 0; i < b.size(); i++) {
		for (int x = 0; x < b[i].size(); x++) {
			b[i][x] -= (dB[i][x] * learning_rate);
		}
	}

	return std::make_tuple(w, b);
}

float VNN_Accuracy(Matrix last_activation, std::vector<float> labels) {
	// Calculate accuracy
	std::vector<float> predictions = std::vector<float>(last_activation.ColumnCount);
	int correct = 0;

	for (int i = 0; i < last_activation.ColumnCount; i++) {

		predictions[i] = last_activation.Column(i)[0];
		predictions[i] > 0.5f ? predictions[i] = 1 : predictions[i] = 0;

		if (predictions[i] == labels[i]) { correct++; }
	}
	return (float)correct / labels.size();
}

std::tuple<Matrix, std::vector<float>> ShuffleInput(Matrix in, std::vector<float> labels) {

	// Shuffle input
	for (int k = 0; k < in.ColumnCount; k++) {

		int r = k + rand() % (in.ColumnCount - k);

		std::vector<float> tempI = in.Column(k);
		int tempL = labels[k];

		in.SetColumn(k, in.Column(r));
		labels[k] = labels[r];

		in.SetColumn(r, tempI);
		labels[r] = tempL;
	}

	return std::make_tuple(in, labels);
}



