#include <map>
#include <cstdlib>
#include <vector>
#include <fstream>
#include "json.hpp"
std::map<char, int> charClassification = {
    // Lowercase letters
    {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5}, {'g', 6},
    {'h', 7}, {'i', 8}, {'j', 9}, {'k', 10}, {'l', 11}, {'m', 12}, {'n', 13},
    {'o', 14}, {'p', 15}, {'q', 16}, {'r', 17}, {'s', 18}, {'t', 19}, {'u', 20},
    {'v', 21}, {'w', 22}, {'x', 23}, {'y', 24}, {'z', 25},

    // Uppercase letters
    {'A', 26}, {'B', 27}, {'C', 28}, {'D', 29}, {'E', 30}, {'F', 31}, {'G', 32},
    {'H', 33}, {'I', 34}, {'J', 35}, {'K', 36}, {'L', 37}, {'M', 38}, {'N', 39},
    {'O', 40}, {'P', 41}, {'Q', 42}, {'R', 43}, {'S', 44}, {'T', 45}, {'U', 46},
    {'V', 47}, {'W', 48}, {'X', 49}, {'Y', 50}, {'Z', 51},

    // Digits
    {'0', 52}, {'1', 53}, {'2', 54}, {'3', 55}, {'4', 56},
    {'5', 57}, {'6', 58}, {'7', 59}, {'8', 60}, {'9', 61},

    // Whitespace
    {' ', 62}, {'\t', 63}, {'\n', 64},

    // Punctuation & symbols
    {'!', 65}, {'"', 66}, {'#', 67}, {'$', 68}, {'%', 69}, {'&', 70},
    {'\'', 71}, {'(', 72}, {')', 73}, {'*', 74}, {'+', 75}, {',', 76},
    {'-', 77}, {'.', 78}, {'/', 79}, {':', 80}, {';', 81}, {'<', 82},
    {'=', 83}, {'>', 84}, {'?', 85}, {'@', 86}, {'[', 87}, {'\\', 88},
    {']', 89}, {'^', 90}, {'_', 91}, {'`', 92}, {'{', 93}, {'|', 94},
    {'}', 95}, {'~', 96}
};
struct DatasetExtract {
    std::vector<int> input;
    float label;
};
using json = nlohmann::json;
DatasetExtract getExampleFromFile(const std::string& filename, std::size_t targetIndex) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse JSON: ") + e.what());
    }

    if (!j.contains("dataset") || !j["dataset"].is_array()) {
        throw std::runtime_error("JSON missing 'dataset' array");
    }

    const auto& arr = j["dataset"];
    if (targetIndex >= arr.size()) {
        throw std::out_of_range("Index out of range");
    }

    const auto& entry = arr[targetIndex];
    if (!entry.contains("input") || !entry.contains("label")) {
        throw std::runtime_error("Entry missing 'input' or 'label'");
    }

    DatasetExtract ex;
    ex.input = entry["input"].get<std::vector<int>>();
    ex.label = entry["label"].get<float>();
    return ex;
}
float train(int enumCount, float changeSize, int enumSize, float bias) {
    float weights[20];
    for (float weight : weights) {
        weight = 50.0f;
    }
    for (int i = {0}; i < enumCount; i++) {
        DatasetExtract trainingExample[enumSize];
        for (int ii = {0}; ii < enumSize; ii++) {
            trainingExample[ii] = getExampleFromFile("rust_detection_dataset.json", (rand() % 501));
        }
        float z = () + 
    }

}
