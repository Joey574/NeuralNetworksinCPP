#include <iostream>
#include <complex>

#include "NeuralNetwork.h"

// Feature Engineering
int fourierSeries = 32;
int chebyshevSeries = 0;
int taylorSeries = 0;
int legendreSeries = 0;
int laguerreSeries = 0;

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

// Prototypes
float mandlebrot(float x, float y, int maxIterations);
std::tuple<Matrix, Matrix> MakeDataSet(int size, int iterations = 50);

int main()
{
	NeuralNetwork model = NeuralNetwork();

	std::vector<int> dims = { 2, 32, 1 };
	std::unordered_set<int> res = { };
	std::unordered_set<int> batch_norm = { };

	NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::loss_metrics eval_metric = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	Matrix::init init_tech = Matrix::init::He;
		
	int batch_size = 264;
	int epochs = 10;
	float valid_split = 0.0f;
	int valid_freq = 5;

    Matrix x;
    Matrix y;

    std::tie(x, y) = MakeDataSet(10000);
    dims[0] = x.RowCount;

	model.Define(dims, res, batch_norm, &Matrix::_ELU, &Matrix::_ELUDerivative, &Matrix::Sigmoid);

	model.Compile(loss, eval_metric, optimizer, init_tech);

	model.Fit(x, y, batch_size, epochs, valid_split, true, valid_freq);
}

float mandlebrot(float x, float y, int maxIterations) {
    std::complex<double> c(x, y);
    std::complex<double> z = 0;

    for (int i = 0; i < maxIterations; ++i) {
        z = z * z + c;
        if (std::abs(z) > 2) {
            return 1.0f - (1.0f / (((float)i / 50.0f) + 1.0f)); // Point is outside + smooth
        }
    }
    return 1.0f; // Point is inside the Mandelbrot set
}

std::tuple<Matrix, Matrix> MakeDataSet(int size, int iterations) {

    Matrix input;
    Matrix labels;

    float xMin = -2.5f;
    float xMax = 1.0f;

    float yMin = -1.1f;
    float yMax = 1.1f;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> xRand(xMin, xMax);
    std::uniform_real_distribution<float> yRand(yMin, yMax);

    input = Matrix(2, size);
    labels = Matrix(1, size);

    for (int i = 0; i < size; i++) {
        float x = xRand(gen);
        float y = yRand(gen);

        float mandle = mandlebrot(x, y, iterations);

        input.SetColumn(i, std::vector<float> {x, y});
        labels.SetColumn(i, std::vector<float> {mandle});
    }

    input = input.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries,
        laguerreSeries, lowerNormalized, upperNormalized);

    return std::make_tuple(input, labels);
}