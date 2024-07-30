#define NOMINMAX
#define _USE_MATH_DEFINES
#include <iostream>
#include <atlimage.h>
#include <complex>

#include "NeuralNetworkTest.h"

// Feature Engineering
int fourierSeries = 0;
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

    std::vector<int> dims = { 2, 10, 10, 1 };
    std::unordered_set<int> res = { };
    std::unordered_set<int> batch_norm = { };

    NeuralNetwork::loss_metrics loss = NeuralNetwork::loss_metrics::mae;
    NeuralNetwork::loss_metrics eval_metric = NeuralNetwork::loss_metrics::mae;
    NeuralNetwork::optimization_technique optimizer = NeuralNetwork::optimization_technique::none;
    Matrix::init init_tech = Matrix::init::He;

    int batch_size = 8;
    int epochs = 1;
    float learning_rate = 0.1f;
    float valid_split = 0.05f;
    int valid_freq = 5;

    Matrix x;
    Matrix y;

    int image_width = 800;
    int image_height = 450;

    model.Define(dims, res, batch_norm, &Matrix::ReLU, &Matrix::ReLUDerivative, &Matrix::ReLU);
    model.Compile(loss, eval_metric, optimizer, init_tech);

    x = Matrix({
        {-2.5f, 0.0f},
        {-2.0f, 0.0f},
        {-1.5f, 0.0f},
        {-1.0f, 0.0f},
        {-0.5f, 0.0f},
        {0.0f, 0.0f},
        {0.5f, 0.0f},
        {1.0f, 0.0f}
        });

    y = Matrix({
        {mandlebrot(x[0][0], x[0][1], 50)},
        {mandlebrot(x[1][0], x[1][1], 50)},
        {mandlebrot(x[2][0], x[2][1], 50)},
        {mandlebrot(x[3][0], x[3][1], 50)},
        {mandlebrot(x[4][0], x[4][1], 50)},
        {mandlebrot(x[5][0], x[5][1], 50)},
        {mandlebrot(x[6][0], x[6][1], 50)},
        {mandlebrot(x[7][0], x[7][1], 50)},
        });

    model.Fit(x, y, batch_size, epochs, learning_rate, valid_split, false, valid_freq);
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

    Matrix image = Matrix(2, width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<float> val = { xMin + (float)x * scaleX, yMin + (float)y * scaleY };
            image.SetColumn(x + (y * width), val);
        }
    }

    image = image.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries,
        laguerreSeries, lowerNormalized, upperNormalized).Transpose();

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