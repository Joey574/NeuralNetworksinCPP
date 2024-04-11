#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

#define _USE_MATH_DEFINES
#include <iostream>
#include <complex>
#include <random>
#include <chrono>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <windows.h>
#include <thread>
#include <iomanip>
#include <cmath>
#include <atlimage.h>

#include "Matrix.h"
#include "ActivationFunctions.h"

// Hyperparameters
std::vector<int> dimensions = { 2, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 1 };
std::unordered_set<int> resNet = { 1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17 };
int fourierSeries = 256;

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

Matrix::init initType = Matrix::init::He;
int epochs = 1500;
int batchSize = 500;
float learningRate = 0.01;

// Inputs
Matrix input;
Matrix batch;

std::vector<float> inputLabels;
std::vector<float> batchLabels;

// Neural Network Matrices
std::vector<Matrix> weights;
std::vector<std::vector<float>> biases;

std::vector<Matrix> activation;
std::vector<Matrix> aTotal;

std::vector<Matrix> dTotal;
std::vector<Matrix> dWeights;
std::vector<std::vector<float>> dBiases;

// Image stuff
Matrix image;
int imageWidth = 1200;
int imageHeight = 700;

// Prototypes
std::wstring NarrowToWide(const std::string& narrowStr);
float mandlebrot(float x, float y, int maxIterations = 50);
void MakeDataSet(int size);
void MakeBMP(std::string filename);
void ForwardPropogation(Matrix in);
void BackwardPropogation();
void ShuffleInput();
Matrix GetNextInput(Matrix totalInput, int size, int i);
void InitializeNetwork();
void InitializeResultMatrices(int size);
float Accuracy(std::vector<int> predictions, std::vector<int> labels);
void CleanTime(double time);
void TrainNetwork();
void UpdateNetwork();


int main()
{
    srand(time(0));

    MakeDataSet(20000);

    InitializeNetwork();

    TrainNetwork();
}

void ShuffleInput() {
    for (int k = 0; k < input.ColumnCount; k++) {

        int r = k + rand() % (input.ColumnCount - k);

        std::vector<float> tempI = input.Column(k);
        int tempL = inputLabels[k];

        input.SetColumn(k, input.Column(r));
        inputLabels[k] = inputLabels[r];

        input.SetColumn(r, tempI);
        inputLabels[r] = tempL;
    }
}

Matrix GetNextInput(Matrix totalInput, int size, int i) {
    Matrix a = Matrix(totalInput.RowCount, size);

    a = totalInput.SegmentC(i * size, i * size + size);
    batchLabels.clear();

    for (int x = i * size; x < i * size + size; x++) {
        batchLabels.push_back(inputLabels[x]);
    }

    return a;
}

void InitializeNetwork() {

    auto initStart = std::chrono::high_resolution_clock::now();

    int connections = 0;

    weights.clear();
    dWeights.clear();
    biases.clear();
    dBiases.clear();

    weights.reserve(dimensions.size() - 1);
    dWeights.reserve(dimensions.size() - 1);

    biases.reserve(dimensions.size() - 1);
    dBiases.reserve(dimensions.size() - 1);

    for (int i = 0; i < dimensions.size() - 1; i++) {
        if (resNet.find(i - 1) != resNet.end()) {
            weights.emplace_back(dimensions[i] + dimensions[0], dimensions[i + 1], initType);
        }
        else {
            weights.emplace_back(dimensions[i], dimensions[i + 1], initType);
        }
        std::cout << "Weights[" << i << "] connections: " << (weights[i].ColumnCount * weights[i].RowCount) << std::endl;
        connections += weights[i].ColumnCount * weights[i].RowCount;

        biases.emplace_back(std::vector<float>(dimensions[i + 1], 0));

        std::cout << "Biases[" << i << "] connections: " << biases[i].size() << std::endl;
        connections += biases[i].size();

        dWeights.emplace_back(weights[i].RowCount, weights[i].ColumnCount);
        dBiases.emplace_back(biases[i].size());
    }

    double fileSize = ((sizeof(float)) * (connections)) + ((sizeof(int) * weights.size() + 1));

    std::cout << "Total connections: " << connections << std::endl;
    std::cout << "Predicted size of file: " << (fileSize / 1000000.00) << "mb" << std::endl;

    aTotal.reserve(weights.size());
    activation.reserve(weights.size());
    dTotal.reserve(weights.size());

    for (int i = 0; i < weights.size(); i++) {
        if (resNet.find(i) != resNet.end()) {
            aTotal.emplace_back(weights[i].ColumnCount + dimensions[0], batchSize);
        }
        else {
            aTotal.emplace_back(weights[i].ColumnCount, batchSize);
        }
        activation.emplace_back(aTotal[i].RowCount, batchSize);
        dTotal.emplace_back(aTotal[i].RowCount, batchSize);
    }

    auto initEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> initTime = initEnd - initStart;
    std::cout << "INITIALIZATION COMPLETE (" << initTime.count() << "ms)" << std::endl;
}

void InitializeResultMatrices(int size) {
    aTotal.clear();
    activation.clear();

    aTotal.reserve(weights.size());
    activation.reserve(weights.size());

    for (int i = 0; i < weights.size(); i++) {
        if (resNet.find(i) != resNet.end()) {
            aTotal.emplace_back(weights[i].ColumnCount + dimensions[0], size);
        }
        else {
            aTotal.emplace_back(weights[i].ColumnCount, size);
        }

        activation.emplace_back(aTotal[i].RowCount, size);
    }
}

float Accuracy(std::vector<float> predictions, std::vector<int> labels) {
    int correct = 0;

    for (int i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == labels[i])
        {
            correct++;
        }
    }
    return (float)correct / (float)predictions.size();
}

void CleanTime(double time) {
    const double hour = 3600000.00;
    const double minute = 60000.00;
    const double second = 1000.00;

    if (time / hour > 1.00) {
        std::cout << time / hour << " hours";
    }
    else if (time / minute > 1.00) {
        std::cout << time / minute << " minutes";
    }
    else if (time / second > 1.00) {
        std::cout << time / second << " seconds";
    }
    else {
        std::cout << time << " ms";
    }
    std::cout << std::endl;
}

std::wstring NarrowToWide(const std::string& narrowStr) {
    int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, nullptr, 0);
    std::wstring wideStr(wideStrLength, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, narrowStr.c_str(), -1, &wideStr[0], wideStrLength);
    return wideStr;
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

void MakeDataSet(int size) {
    float xMin = -2.5f;
    float xMax = 1.0f;

    float yMin = -1.1f;
    float yMax = 1.1f;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> xRand(xMin, xMax);
    std::uniform_real_distribution<float> yRand(yMin, yMax);

    input = Matrix(2, size);

    for (int i = 0; i < size; i++) {
        float x = xRand(gen);
        float y = yRand(gen);

        input.SetColumn(i, std::vector<float> {x, y});
        float mandle = mandlebrot(x, y);
        inputLabels.push_back(mandle);
    }

   image = Matrix(2, imageWidth * imageHeight);

   float scaleX = (std::abs(xMin - xMax)) / (imageWidth - 1);
   float scaleY = (std::abs(yMin - yMax)) / (imageHeight - 1);

    for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
            std::vector<float> val = { xMin + (float)x * scaleX, yMin + (float)y * scaleY };
            image.SetColumn(x + (y * imageWidth), val);
        }
    }

    // Normalize Input
    input = input.Normalized(lowerNormalized, upperNormalized);
    image = image.Normalized(lowerNormalized, upperNormalized);

    Matrix oldI = input;
    Matrix oldF = image;
    // Compute Fourier Series
    if (fourierSeries > 0) {
        auto sTime = std::chrono::high_resolution_clock::now();
        std::cout << "Computing " << fourierSeries << " order(s) of Fourier Series..." << std::endl;

        for (int f = 0; f < fourierSeries; f++) {
            input = input.Combine(oldI.FourierSeries(f + 1));
            image = image.Combine(oldF.FourierSeries(f + 1));
        }
        dimensions[0] = input.RowCount;

        std::cout << "Fourier Features: " << input.RowCount - oldI.RowCount << std::endl;

        std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - sTime;
        std::cout << "Time to compute " << fourierSeries << " order(s): " << time.count() / 1000.00 << " seconds" << std::endl;
    }
}

void MakeBMP(std::string filename) {
    std::wstring fp = NarrowToWide(filename);

    CImage mandlebrot;

    mandlebrot.Create(imageWidth, imageHeight, 24);

    InitializeResultMatrices(image.ColumnCount);
    ForwardPropogation(image);

    std::cout << "Image FP completed..." << std::endl;

    std::vector<float> pixelsData = activation[activation.size() - 1].Row(0);

    for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
            mandlebrot.SetPixel(x, y, RGB(pixelsData[x + (y * imageWidth)] * 255, 0, 0));
        }
    }

    InitializeResultMatrices(batchSize);

    mandlebrot.Save(fp.c_str(), Gdiplus::ImageFormatBMP);
}


void TrainNetwork() {
    std::cout << "TRAINING STARTED" << std::endl;

    std::chrono::steady_clock::time_point totalStart;
    std::chrono::steady_clock::time_point tStart;
    std::chrono::duration<double, std::milli> time;
    std::chrono::duration<double, std::milli> timeToReachHighest;

    totalStart = std::chrono::high_resolution_clock::now();

    int iterations = input.ColumnCount / batchSize;

    std::cout << std::fixed << std::setprecision(4);
    for (int e = 0; e < epochs; e++) {

        tStart = std::chrono::high_resolution_clock::now();

        ShuffleInput();
        for (int i = 0; i < iterations; i++) {

            batch = GetNextInput(input, batchSize, i);

            ForwardPropogation(batch);
            BackwardPropogation();
            UpdateNetwork();
        }

        if (e % 10 == 9) {
            std::string filename = "MandlebrotAproximations\\" + std::to_string(e).append(".bmp");
            MakeBMP(filename);
            InitializeResultMatrices(batchSize);
        }

        time = std::chrono::high_resolution_clock::now() - tStart;
        std::cout << "Epoch: " << e << " Epoch Time: ";
        CleanTime(time.count());
    }

    time = (std::chrono::high_resolution_clock::now() - totalStart);
    float epochTime = time.count() / epochs;

    std::cout << "Total Training Time: ";
    CleanTime(time.count());
    std::cout << "Average Epoch Time: ";
    CleanTime(epochTime);

    tStart = std::chrono::high_resolution_clock::now();
    std::string filename = "MandlebrotAproximations\\MandlebrotAprox.bmp";
    MakeBMP(filename);
    time = (std::chrono::high_resolution_clock::now() - tStart);

    std::cout << "Image Made: " << std::endl;
    CleanTime(time.count());
}

void ForwardPropogation(Matrix in) {

    for (int i = 0; i < aTotal.size(); i++) {
        if (resNet.find(i) != resNet.end()) {

            aTotal[i].Insert(0, in);
            activation[i].Insert(0, in);

            aTotal[i].Insert(in.RowCount, (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose());
        }
        else {
            aTotal[i] = (weights[i].DotProduct(i == 0 ? in : activation[i - 1]) + biases[i]).Transpose();
        }
        activation[i] = i < aTotal.size() - 1 ? LeakyReLU(aTotal[i]) : Sigmoid(aTotal[i]);
    }
}

void BackwardPropogation() {

    dTotal[dTotal.size() - 1] = activation[activation.size() - 1] - batchLabels;

    for (int i = dTotal.size() - 2; i > -1; i--) {

        if (resNet.find(i) != resNet.end()) {
            dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1].SegmentR(batch.RowCount))).Transpose() * LeakyReLUDerivative(aTotal[i].SegmentR(batch.RowCount)));
        }
        else {
            dTotal[i] = ((dTotal[i + 1].DotProduct(weights[i + 1])).Transpose() * LeakyReLUDerivative(aTotal[i]));
        }
    }

    std::for_each(std::execution::par_unseq, dWeights.begin(), dWeights.end(), [&](auto&& item) {
        size_t i = &item - dWeights.data();
        item = (dTotal[i].Transpose().DotProduct(i == 0 ? batch.Transpose() : activation[i - 1].Transpose()) * (1.0f / (float)batchSize)).Transpose();
        dBiases[i] = dTotal[i].Multiply(1.0f / (float)batchSize).RowSums();
        });
}

void UpdateNetwork() {
    for (int i = 0; i < weights.size(); i++) {
        weights[i] -= dWeights[i].Multiply(learningRate);
    }

    for (int i = 0; i < biases.size(); i++) {
        for (int x = 0; x < biases[i].size(); x++) {
            biases[i][x] -= (dBiases[i][x] * learningRate);
        }
    }
}
