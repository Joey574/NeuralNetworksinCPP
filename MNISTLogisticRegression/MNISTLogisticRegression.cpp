//#include <iostream>
//#include <vector>
//#include <execution>
//#include <unordered_set>
//#include <algorithm>
//#include <random>
//#include <armadillo>
//
//#include "ActivationFunctions.cpp"
//
//using namespace arma;
//using namespace std;
//
//// Hyperparameters
//int inputSize;
//int outputSize;
//vector<int> hiddenSize;
//
//int iterations;
//int batchSize;
//
//float learningRate;
//float thresholdAccuracy;
//
//// Theta Parameters
//vector<fmat> weights;
//fmat biases;
//
//// Theta derivs
//vector<fmat> dTotal;
//vector<fmat> dWeights;
//fmat dBiases;
//
//// Output / Input
//fmat input;
//vector<int> inputLabels;
//fmat batch;
//vector<int> batchLabels;
//fmat Y;
//fmat YComplete;
//vector<fmat> A;
//vector<fmat> ATotal;
//
//// Protos
//void UpdateNetwork();
//void ForwardPropogation(fmat in);
//void BackwardPropogation();
//float Accuracy(vector<float> predictions, vector<int> labls);
//vector<float> GetPredictions(int len);
//
//
//
int main()
{

}
//
//fmat RandomizeInput(fmat in, int batchNum) {
//	unordered_set<int> used = unordered_set<int>(batchNum);
//	fmat a = fmat(input.n_rows, batchNum);
//	batchLabels.clear();
//	Y.clear();
//
//	for (int i = 0; i < batchNum; i++) {
//		int c = rand() % in.n_cols;
//
//		if (!used.count(c)) {
//			used.insert(c);
//
//			a.col(batchLabels.size()) = input.col(c);
//			Y.col(batchLabels.size()) = YComplete(c);
//
//			batchLabels.push_back(inputLabels[c]);
//		}
//	}
//	return a;
//}
//
//void InitializeNetwork() {
//
//}
//
//void TrainNetwork() {
//
//	batch = RandomizeInput(input, batchSize);
//
//	for (int i = 0; i < iterations; i++) {
//		float acc = Accuracy(GetPredictions(batchSize), batchLabels);
//
//		if (acc > thresholdAccuracy) {
//			batch = RandomizeInput(input, batchSize);
//		}
//
//		cout << "Iteration: " << i << " Accuracy: " << acc << endl;
//
//		ForwardPropogation(batch);
//
//		BackwardPropogation();
//
//		UpdateNetwork();
//	}
//}
//
//void TestNetwork() {
//
//}
//
//void ForwardPropogation(fmat in) {
//	for (int i = 0; i < A.size(); i++) {
//		/*for (int c = 0; c < A[i].n_cols; c++) {
//			ATotal[i].col(c) = weights[i] * (i == 0 ? batch.col(c) : A[i - 1].col(c)) + biases[i];
//		}*/
//		ATotal[i] = weights[i] * (i == 0 ? batch : A[i - 1]) + biases[i];
//		A[i] = i < A.size() - 1 ? ReLU(ATotal[i]) : Softmax(ATotal[i]);
//	}
//}
//
//void BackwardPropogation() {
//	dTotal[dTotal.size() - 1] -= Y;
//}
//
//void UpdateNetwork() {
//
//}
//
//vector<float> GetPredictions(int len) {
//	vector<float> predictions;
//	
//	for (int i = 0; i < len; i++) {
//		predictions.push_back(A[A.size() - 1].col(i).index_max());
//	}
//	return predictions;
//}
//
//float Accuracy(vector<float> predictions, vector<int> labls) {
//	int correct = 0;
//
//	for (int i = 0; i < predictions.size(); i++) {
//		if ((int)(predictions[i] + 0.0001f) == labls[i]) {
//			correct++;
//		}
//	}
//	return (float)correct / predictions.size();
//}
//
//
//
