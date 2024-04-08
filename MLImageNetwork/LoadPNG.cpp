#include "LoadPNG.h"

bool getBit(unsigned char byte, int position) {
    return (byte >> position) & 1;
}

void LoadPNG(std::string filename) {
    std::ifstream file = std::ifstream(filename, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        std::cout << "File not found" << std::endl;
    } else {
        std::cout << "Loading image..." << std::endl;
    }

    char c;

    file.read(reinterpret_cast<char*>(&c), sizeof(c));
    std::cout << c;
    file.read(reinterpret_cast<char*>(&c), sizeof(c));
    std::cout << c << std::endl;

    int32_t size;

    // File size
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::cout << size << std::endl;

    // Reserved
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    int32_t offset;
    file.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    std::cout << offset << std::endl;

    // above = 14 bytes


    // DIB size
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::cout << size << std::endl;

    // width
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::cout << size << std::endl;
    // height
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::cout << size << std::endl;

    // 26 bytes
}