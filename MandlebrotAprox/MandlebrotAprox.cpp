#pragma comment(linker, "/STACK:20000000")
#pragma comment(linker, "/HEAP:20000000")

#define NOMINMAX
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
#include <algorithm>

#include "Matrix.h"
#include "ActivationFunctions.h"

// Hyperparameters
std::vector<int> dimensions = { 2, 32, 32, 1 };
std::unordered_set<int> resNet = {  };
std::unordered_set<int> batch_normalization = {  };

float lowerNormalized = -M_PI;
float upperNormalized = M_PI;

Matrix::init initType = Matrix::init::He;
int epochs = 5;
int batchSize = 500;
float learningRate = 0.01f;

// Feature Engineering
int fourierSeries = 12;
int chebyshevSeries = 0;
int taylorSeries = 0;
int legendreSeries = 0;
int laguerreSeries = 0;

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

// Save / Load
bool SaveOnComplete = false;
bool LoadOnInit = false;
std::string NetworkPath = "22_150_256_0_0_0_0.txt";

// Image stuff / Mandlebrot specific
int dataSize = 20000;
int mandlebrotIterations = 500;
int epochPerDataset = 10;
int epochPerImage = 10;

std::vector<Matrix> imageVector;
int imageWidth = 160;
int imageHeight = 90;

int finalWidth = 160;
int finalHeight = 90;

int cacheSize = (4 * 1000000);
int pixelPerMatrix;

/*
Common Resolutions:

16:9
    160 x 90
    320 x 180
    800 x 450
    1920 x 1080
    2560 x 1440
    3840 x 2160
    7680 x 4320

16:10
    1920 x 1200
    2560 x 1600
*/

float confidenceThreshold = 0.95f;

// Prototypes
std::wstring NarrowToWide(const std::string& narrowStr);
float mandlebrot(float x, float y, int maxIterations);
void MakeDataSet(int size);
void MakeImageFeatures(int width, int height);
void MakeBMP(std::string filename, int width, int height);
void ForwardPropogation(Matrix in);
void BackwardPropogation();
void ShuffleInput();
Matrix GetNextInput(Matrix totalInput, int size, int i);
void InitializeNetwork();
void InitializeResultMatrices(int size);
void CleanTime(double time);
void TrainNetwork();
void UpdateNetwork();
void SaveNetwork(std::string filename);
void LoadNetwork(std::string filename);
void NetworkToImage(std::string filename);

int main()
{
    srand(time(0));

    MakeDataSet(dataSize);

    if (LoadOnInit) {
        LoadNetwork(NetworkPath);

    }
    else {
        InitializeNetwork();
    }

    MakeImageFeatures(imageWidth, imageHeight);

    TrainNetwork();

    if (SaveOnComplete) { SaveNetwork(NetworkPath); }
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
    std::cout << "Predicted size of file: " << (fileSize / 1280000.00) << "mb" << std::endl;

    aTotal.reserve(weights.size());
    activation.reserve(weights.size());
    dTotal.reserve(weights.size());

    // Initialize Result Matrices
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

void CleanTime(double time) {
    const double hour = 3600000.00;
    const double minute = 60000.00;
    const double second = 1000.00;

    if ((time / hour) > 1.00) {
        std::cout << time / hour << " hours";
    }
    else if ((time / minute) > 1.00) {
        std::cout << time / minute << " minutes";
    }
    else if ((time / second) > 1.00) {
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
    inputLabels.clear();

    for (int i = 0; i < size; i++) {
        float x = xRand(gen);
        float y = yRand(gen);

        input.SetColumn(i, std::vector<float> {x, y});
        float mandle = mandlebrot(x, y, mandlebrotIterations);
        inputLabels.push_back(mandle);
    }

    input = input.ExtractFeatures(fourierSeries, taylorSeries, chebyshevSeries, legendreSeries, 
        laguerreSeries, lowerNormalized, upperNormalized);
    dimensions[0] = input.RowCount;
}

void MakeImageFeatures(int width, int height) {

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
        laguerreSeries, lowerNormalized, upperNormalized);

    image.Transpose();

    int connections = 0;
    for (int i = 0; i < weights.size(); i++) {
        connections += (weights[i].RowCount * weights[i].ColumnCount) + biases[i].size();
    }

    // Reserve size for network itself
    int cacheSizeTemp = cacheSize - (connections * sizeof(float));

    // How many pixels per matrix we can store in cache (pixel size + result matrices)
    pixelPerMatrix = std::max(std::floor((float)cacheSizeTemp / ((image.RowCount + (aTotal[0].RowCount * 2)) * sizeof(float))), 10.0f);

    // Minimum number of matrices we need for the image at optimal number of pixels
    int matrices = std::ceil((float)(width * height) / (float)pixelPerMatrix);

    imageVector = std::vector<Matrix>(matrices);

    for (int i = 0; i < matrices; i++) {
        imageVector[i] = image.SegmentC(i * pixelPerMatrix, std::min((i * pixelPerMatrix) + pixelPerMatrix, image.ColumnCount));
    }

    dimensions[0] = imageVector[0].RowCount;
}

void MakeBMP(std::string filename, int width, int height) {
    std::wstring fp = NarrowToWide(filename);

    CImage mandle;

    mandle.Create(width, height, 24);

    Matrix currentTotal;
    Matrix lastActivation;

    // Pixel index
    int pI = 0;

    //Forward prop on image
    for (int y = 0; y < imageVector.size(); y++) {

        for (int i = 0; i < aTotal.size(); i++) {
            currentTotal = Matrix(aTotal[i].RowCount, imageVector[y].ColumnCount);
            if (resNet.find(i) != resNet.end()) {
                currentTotal.Insert(0, imageVector[y]);
                currentTotal.Insert(imageVector[y].RowCount, (weights[i].DotProduct(i == 0 ? imageVector[y] : lastActivation) + biases[i]).Transpose());
            }
            else {
                currentTotal = (weights[i].DotProduct(i == 0 ? imageVector[y] : lastActivation) + biases[i]).Transpose();
            }
            lastActivation = i < aTotal.size() - 1 ? LeakyReLU(currentTotal) : Sigmoid(currentTotal);
        }

        // Get pixel data
        std::vector<float> pixelData = lastActivation.Row(0);

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


void TrainNetwork() {
    std::cout << "TRAINING STARTED" << std::endl;

    std::chrono::steady_clock::time_point totalStart;
    std::chrono::steady_clock::time_point tStart;
    std::chrono::duration<double, std::milli> time;
    std::chrono::duration<double, std::milli> timeToReachHighest;

    totalStart = std::chrono::high_resolution_clock::now();

    // Get date for file naming
    std::time_t t = std::time(0); std::tm now; localtime_s(&now, &t);
    std::string date = "MandlebrotAproximations\\" + std::to_string(now.tm_mon + 1).append("_").append(std::to_string(now.tm_mday)).append("_")
        .append(std::to_string(now.tm_year - 100));

    int iterations = input.ColumnCount / batchSize;

    std::cout << std::fixed << std::setprecision(4);
    for (int e = 0; e < epochs; e++) {

        tStart = std::chrono::high_resolution_clock::now();

        if (e % epochPerDataset == 0) { MakeDataSet(dataSize); }

        for (int i = 0; i < iterations; i++) {

            batch = GetNextInput(input, batchSize, i);

            ForwardPropogation(batch);
            BackwardPropogation();
            UpdateNetwork();
        }

        if (e % epochPerImage == epochPerImage - 1) {
            std::string filename = (date + "_ep").append(std::to_string(e + 1)).append(".bmp");
            MakeBMP(filename, imageWidth, imageHeight);
            InitializeResultMatrices(batchSize);
        }

        time = std::chrono::high_resolution_clock::now() - tStart;
        std::cout << "Epoch: " << e << " Epoch Time: ";
        CleanTime(time.count());
    }

    time = (std::chrono::high_resolution_clock::now() - totalStart);
    float epochTime = time.count() / epochs;

    std::cout << "Total Training Time: "; CleanTime(time.count());
    std::cout << "Average Epoch Time: "; CleanTime(epochTime);

    tStart = std::chrono::high_resolution_clock::now();
    std::string filename = (date + "_final.bmp");
    MakeImageFeatures(finalWidth, finalHeight);
    MakeBMP(filename, finalWidth, finalHeight);
    time = (std::chrono::high_resolution_clock::now() - tStart);

    std::cout << "Image Made: ";
    CleanTime(time.count());

    NetworkToImage("test.bmp");
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

        if (batch_normalization.find(i) != batch_normalization.end()) {
            activation[i].Normalized(lowerNormalized, upperNormalized);
        }
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


void SaveNetwork(std::string filename) {
    std::ofstream fw = std::ofstream(filename, std::ios::out | std::ios::binary);

    // Number of layers
    int s = dimensions.size();
    fw.write(reinterpret_cast<const char*>(&s), sizeof(int));

    // Number of resNet
    int r = resNet.size();
    fw.write(reinterpret_cast<const char*>(&r), sizeof(int));

    // Write dimensions of network
    fw.write(reinterpret_cast<const char*>(dimensions.data()), dimensions.size() * sizeof(int));

    // Write resNet layers
    std::vector<int> res;
    for (auto it = resNet.begin(); it != resNet.end(); ) {
        res.push_back(std::move(resNet.extract(it++).value()));
    }
    fw.write(reinterpret_cast<const char*>(res.data()), res.size() * sizeof(int));

    // Write feature engineering stuff
    std::vector<int> features = { fourierSeries, taylorSeries, chebyshevSeries, legendreSeries, laguerreSeries };
    fw.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(int));

    // Write weights
    for (int i = 0; i < weights.size(); i++) {
        for (int r = 0; r < weights[i].RowCount; r++) {
            fw.write(reinterpret_cast<const char*>(weights[i].Row(r).data()), weights[i].Row(r).size() * sizeof(float));
        }
    }

    // Write biases
    for (int i = 0; i < biases.size(); i++) {
        fw.write(reinterpret_cast<const char*>(biases[i].data()), biases[i].size() * sizeof(float));
    }

    fw.close();

    std::cout << "NETWORK SAVED" << std::endl;
}

void LoadNetwork(std::string filename) {
    std::ifstream fr = std::ifstream(filename, std::ios::in | std::ios::binary);

    if (fr.is_open()) {
        std::cout << "Loading Network..." << std::endl;
    }
    else {
        std::cout << "Network not found..." << std::endl;
    }

    // Network size
    int s;
    fr.read(reinterpret_cast<char*>(&s), sizeof(int));
    std::cout << "Network Layers: " << s << std::endl;

    // ResNet size
    int r;
    fr.read(reinterpret_cast<char*>(&r), sizeof(int));
    std::cout << "Resnet Layers: " << r << std::endl;

    // Read dimensions
    dimensions = std::vector<int>(s);
    fr.read(reinterpret_cast<char*>(dimensions.data()), s * sizeof(int));
    std::cout << "Layer Size: " << dimensions[1] << std::endl;

    // Read resNet
    resNet.clear();
    std::vector<int> res = std::vector<int>(r);
    fr.read(reinterpret_cast<char*>(res.data()), r * sizeof(int));

    for (int i = 0; i < res.size(); i++) {
        resNet.insert(res[i]);
    }

    // Read feature engineering stuff
    std::vector<int> features = std::vector<int>(5);
    fr.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(int));
    fourierSeries = features[0];
    taylorSeries = features[1];
    chebyshevSeries = features[2];
    legendreSeries = features[3];
    laguerreSeries = features[4];

    InitializeNetwork();

    // Read weights
    for (int i = 0; i < weights.size(); i++) {
        for (int r = 0; r < weights[i].RowCount; r++) {
            std::vector<float> row = std::vector<float>(weights[i].ColumnCount);
            fr.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

            weights[i].SetRow(r, row);
        }
    }

    // Read biases
    for (int i = 0; i < biases.size(); i++) {
        fr.read(reinterpret_cast<char*>(biases[i].data()), biases[i].size() * sizeof(float));
    }

    fr.close();

    std::cout << "NETWORK LOADED" << std::endl;
}


void NetworkToImage(std::string filename) {
    std::wstring fp = NarrowToWide(filename);

    CImage network;

    std::vector<std::vector<float>> normalized_biases = biases;

    // get abs of biases
    for (int i = 0; i < biases.size(); i++) {
        for (int x = 0; x < biases[i].size(); x++) {
            normalized_biases[i][x] = std::abs(biases[i][x]);
        }
    }
    
    // get min and max of biases
    std::vector<float> b_max_vec;
    std::vector<float> b_min_vec;
    for (int b = 0; b < biases.size(); b++) {
        b_max_vec.push_back(*std::max_element(normalized_biases[b].begin(), normalized_biases[b].end()));
        b_min_vec.push_back(*std::min_element(normalized_biases[b].begin(), normalized_biases[b].end()));
    }
    float b_max = *std::min_element(b_max_vec.begin(), b_max_vec.end());
    float b_min = *std::min_element(b_min_vec.begin(), b_min_vec.end());

    // normalize biases to [0, 255]
    for (int i = 0; i < biases.size(); i++) {
        for (int x = 0; x < biases[i].size(); x++) {
            normalized_biases[i][x] = ((normalized_biases[i][x] - b_min) / (b_max - b_min)) * (255);
        }
    }

    std::vector<int> true_dimensions = dimensions;
    for (int i = 0; i < dimensions.size(); i++) {
        if (resNet.find(i - 1) != resNet.end()) {
            true_dimensions[i] += dimensions[0];
        }
    }

    int max = *std::max_element(true_dimensions.begin(), true_dimensions.end());
    int size = true_dimensions.size();

    int circle_diameter = 3;
    int horizontal_gap = 4;
    int vertical_gap = 15;
    int border_size = 4;

    int width = (max * circle_diameter) + ((max - 1) * horizontal_gap) + border_size;
    int height = (size * circle_diameter) + ((size - 1) * vertical_gap) + border_size;
    network.Create(width, height, 24);

    for (int i = 0; i < size; i++) {

        int offset_x = (max - true_dimensions[i]) / 2;

        for (int x = 0; x < true_dimensions[i]; x++) {
            int center_x = ((x + offset_x) * (circle_diameter + horizontal_gap)) + border_size;
            int center_y = (i * (circle_diameter + vertical_gap)) + border_size;
            int c_r = circle_diameter / 2;

            int strength = i == 0 ? 255 : normalized_biases[i - 1][x];

            for (int c_x = -c_r; c_x < c_r; c_x++) {
                for (int c_y = -c_r; c_y < c_r; c_y++) {
                    network.SetPixel(center_x + c_x, center_y + c_y, RGB(strength, strength, strength));
                }
            }
        }
    }

    network.Save(fp.c_str(), Gdiplus::ImageFormatBMP);
    network.Destroy();
}