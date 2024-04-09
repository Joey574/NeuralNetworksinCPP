#include "LoadPNG.h"

bool getBit(unsigned char byte, int position) {
    return (byte >> position) & 1;
}

Matrix LoadPNG(std::string filename) {
    std::ifstream file = std::ifstream(filename, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        std::cout << "File not found" << std::endl;
    } else {
        std::cout << "Loading image..." << std::endl;
    }

    char c;
    // 2 bytes for "BM"
    file.read(reinterpret_cast<char*>(&c), sizeof(c));
    std::cout << c;
    file.read(reinterpret_cast<char*>(&c), sizeof(c));
    std::cout << c << std::endl;

    int32_t temp;
    int32_t offset;
    int32_t w;
    int32_t h;

    // File size
    file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
    // Reserved
    file.read(reinterpret_cast<char*>(&temp), sizeof(temp));

    file.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    std::cout << "Offset: " << offset << std::endl;

    // DIB size
    file.read(reinterpret_cast<char*>(&temp), sizeof(temp));

    // width
    file.read(reinterpret_cast<char*>(&w), sizeof(w));
    std::cout << "Width: " << w << std::endl;
    // height
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    std::cout << "Height: " << h << std::endl;

    // move to offset
    file.seekg(offset);

    uint8_t a;

    std::vector<std::vector<int>> mat;
    std::vector<int> row;

    // Read values into a matrix, these are the labels
    for (int i = 0; i < (w * h) / 8; i++) {
        file.read(reinterpret_cast<char*>(&a), sizeof(a));

        for (int x = 0; x < 8; x++) {
            row.push_back(getBit(a, x));
        }
        if (i % 4 == 3) {
            mat.push_back(row);
            row.clear();
        }
    }

    Matrix input = Matrix(mat.size(), mat[0].size());
    for (int i = 0; i < mat.size(); i++) {
        input.SetRow(i, mat[i]);
    }

    return input;
}