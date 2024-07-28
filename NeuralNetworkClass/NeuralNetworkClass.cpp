#define NOMINMAX
#define _USE_MATH_DEFINES
#include <iostream>
#include <atlimage.h>
#include <complex>

#include "NeuralNetwork.h"

// Feature Engineering
int fourierSeries = 64;
int chebyshevSeries = 0;
int taylorSeries = 0;
int legendreSeries = 0;
int laguerreSeries = 0;

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

int cacheSize = (4 * 1000000);
int pixelPerMatrix;

float confidenceThreshold = 0.95f;

// Prototypes
float mandlebrot(float x, float y, int maxIterations);
std::tuple<Matrix, Matrix> MakeDataSet(int size, int iterations = 50);
std::vector<Matrix>  MakeImageFeatures(int width, int height);
void MakeBMP(std::string filename, int width, int height, std::vector<Matrix> imageVector, NeuralNetwork model);
std::wstring NarrowToWide(const std::string& narrowStr);

int main()
{
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

	NeuralNetwork model = NeuralNetwork();

	std::vector<int> dims = { 2, 32, 32, 1 };
	std::unordered_set<int> res = { };
	std::unordered_set<int> batch_norm = { };

	NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::loss_metrics eval_metric = NeuralNetwork::loss_metrics::mae;
	NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
	Matrix::init init_tech = Matrix::init::He;
		
	int batch_size = 500;
	int epochs = 2;
    float learning_rate = 0.01f;
	float valid_split = 0.05f;
	int valid_freq = 5;

    Matrix x;
    Matrix y;

    int image_width = 800;
    int image_height = 450;

    std::tie(x, y) = MakeDataSet(200000);
    dims[0] = x.ColumnCount;

	model.Define(dims, res, batch_norm, &Matrix::_ELU, &Matrix::_ELUDerivative, &Matrix::Sigmoid);
	model.Compile(loss, eval_metric, optimizer, init_tech);

    for (int i = 0; i < 10; i++) {

        std::tie(x, y) = MakeDataSet(200000);

        model.Fit(x, y, batch_size, epochs, learning_rate, valid_split, true, valid_freq);
        std::vector<Matrix> image_data = MakeImageFeatures(image_width, image_height);

        MakeBMP("test_" + std::to_string(i).append(".bmp"), image_width, image_height, image_data, model);
    }
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
        laguerreSeries, lowerNormalized, upperNormalized).Transpose();
    labels = labels.Transpose();

    return std::make_tuple(input, labels);
}

std::vector<Matrix> MakeImageFeatures(int width, int height) {

    float xMin = -2.5f;
    float xMax = 1.0f;
    float yMin = -1.1f;
    float yMax = 1.1f;

    float scaleX = (std::abs(xMin - xMax)) / (width - 1);
    float scaleY = (std::abs(yMin - yMax)) / (height - 1);

    Matrix image = Matrix(width * height, 2);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<float> val = { xMin + (float)x * scaleX, yMin + (float)y * scaleY };
            image.SetRow(x + (y * width), val);
        }
    }

    image = image.Transpose();
    image = image.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries,
        laguerreSeries, lowerNormalized, upperNormalized);
    image = image.Transpose();

    pixelPerMatrix = width;

    // Minimum number of matrices we need for the image at optimal number of pixels
    int matrices = std::ceil((float)(width * height) / (float)pixelPerMatrix);

    std::vector<Matrix> imageVector = std::vector<Matrix>(matrices);

    // Segment total matrix into vectors
    for (int i = 0; i < matrices; i++) {
        imageVector[i] = image.SegmentR(i * pixelPerMatrix, std::min((i * pixelPerMatrix) + pixelPerMatrix, image.RowCount));
    }

    return imageVector;
}

void MakeBMP(std::string filename, int width, int height, std::vector<Matrix> imageVector, NeuralNetwork model) {
    std::wstring fp = NarrowToWide(filename);

    CImage mandle;

    mandle.Create(width, height, 24);

    Matrix currentTotal;
    Matrix lastActivation;

    // Pixel index
    int pI = 0;

    //Forward prop on image
    for (int y = 0; y < imageVector.size(); y++) {

        // Get pixel data
        std::vector<float> pixelData = model.Predict(imageVector[y]).Column(0);

        // Set pixels
        for (int x = 0; x < pixelData.size() && pI < width * height; x++) {
            float r = pixelData[x] * 255.0f;
            float other = pixelData[x] > confidenceThreshold ? 255 : 0;

            mandle.SetPixel(pI % width, pI / width, RGB(r, other, other));
            pI++;
        }
    }

    mandle.Save(fp.c_str(), Gdiplus::ImageFormatBMP);
    mandle.Destroy();
}

std::wstring NarrowToWide(const std::string& narrowStr) {
    int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, nullptr, 0);
    std::wstring wideStr(wideStrLength, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, &wideStr[0], wideStrLength);
    return wideStr;
}
