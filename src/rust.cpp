#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>
#include <filesystem>
#include <curl/curl.h>

#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ------------------ Character Map ------------------
std::map<char,int> charClassification = {
    {'a',0},{'b',1},{'c',2},{'d',3},{'e',4},{'f',5},{'g',6},{'h',7},{'i',8},{'j',9},
    {'k',10},{'l',11},{'m',12},{'n',13},{'o',14},{'p',15},{'q',16},{'r',17},{'s',18},
    {'t',19},{'u',20},{'v',21},{'w',22},{'x',23},{'y',24},{'z',25},
    {'A',26},{'B',27},{'C',28},{'D',29},{'E',30},{'F',31},{'G',32},{'H',33},{'I',34},
    {'J',35},{'K',36},{'L',37},{'M',38},{'N',39},{'O',40},{'P',41},{'Q',42},{'R',43},
    {'S',44},{'T',45},{'U',46},{'V',47},{'W',48},{'X',49},{'Y',50},{'Z',51},
    {'0',52},{'1',53},{'2',54},{'3',55},{'4',56},{'5',57},{'6',58},{'7',59},{'8',60},{'9',61},
    {' ',62},{'\t',63},{'\n',64},{'!',65},{'"',66},{'#',67},{'$',68},{'%',69},{'&',70},
    {'\'',71},{'(',72},{')',73},{'*',74},{'+',75},{',',76},{'-',77},{'.',78},{'/',79},
    {':',80},{';',81},{'<',82},{'=',83},{'>',84},{'?',85},{'@',86},{'[',87},{'\\',88},
    {']',89},{'^',90},{'_',91},{'`',92},{'{',93},{'|',94},{'}',95},{'~',96}
};

const int CHAR_COUNT = 97;
const int MAX_INPUT_LEN = 20;

// ------------------ Dataset Struct ------------------
struct DatasetExtract {
    std::vector<int> input;
    float label;
};

// ------------------ Curl download ------------------
size_t curlWrite(void* ptr, size_t size, size_t nmemb, std::string* data) {
    data->append((char*)ptr, size * nmemb);
    return size * nmemb;
}

void downloadDataset(const std::string& url, const std::string& outFile) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("Failed to init CURL");

    std::string buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("CURL download failed: " + std::string(curl_easy_strerror(res)));
    }
    curl_easy_cleanup(curl);

    std::ofstream f(outFile);
    f << buffer;
}

// ------------------ Load Dataset ------------------
std::vector<DatasetExtract> loadDataset(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    json j;
    file >> j;
    if (!j.contains("dataset") || !j["dataset"].is_array())
        throw std::runtime_error("JSON missing 'dataset' array");

    std::vector<DatasetExtract> dataset;
    for (const auto& entry : j["dataset"]) {
        if (!entry.contains("input") || !entry.contains("label")) continue;
        DatasetExtract ex;
        ex.input = entry["input"].get<std::vector<int>>();
        ex.label = entry["label"].get<float>();
        dataset.push_back(ex);
    }
    return dataset;
}

// ------------------ One-hot encode ------------------
std::vector<float> oneHotEncode(const std::vector<int>& input) {
    std::vector<float> encoded(MAX_INPUT_LEN * CHAR_COUNT, 0.0f);
    for (int i = 0; i < MAX_INPUT_LEN && i < input.size(); ++i) {
        int idx = input[i];
        if (idx >= 0 && idx < CHAR_COUNT) encoded[i * CHAR_COUNT + idx] = 1.0f;
    }
    return encoded;
}

// ------------------ Activation ------------------
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float dsigmoid(float y) { return y * (1.0f - y); }

// ------------------ NeuralNet ------------------
struct NeuralNet {
    int inputSize;
    int hiddenSize;
    float learningRate;
    std::vector<float> W1, b1, W2;
    float b2;

    NeuralNet(int inSize, int hidSize, float lr)
        : inputSize(inSize), hiddenSize(hidSize), learningRate(lr), b2(0.0f)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.01f,0.01f);

        W1.resize(inputSize * hiddenSize);
        b1.resize(hiddenSize, 0.0f);
        W2.resize(hiddenSize);

        for (auto &w : W1) w = dist(gen);
        for (auto &w : W2) w = dist(gen);
    }

    float forward(const std::vector<float>& x, std::vector<float>& hiddenOut) const {
        hiddenOut.resize(hiddenSize);
        for (int j = 0; j < hiddenSize; ++j) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; ++i) sum += x[i] * W1[i*hiddenSize + j];
            hiddenOut[j] = sigmoid(sum);
        }
        float z = b2;
        for (int j = 0; j < hiddenSize; ++j) z += hiddenOut[j] * W2[j];
        return sigmoid(z);
    }

    void train(const std::vector<DatasetExtract>& dataset, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epochLoss = 0.0f;
            std::vector<DatasetExtract> data = dataset;
            std::mt19937 g(std::random_device{}());
            std::shuffle(data.begin(), data.end(), g);

            for (auto &ex : data) {
                auto x = oneHotEncode(ex.input);
                std::vector<float> hidden;
                float y = forward(x, hidden);
                float error = y - ex.label;
                epochLoss += -(ex.label * std::log(y + 1e-7f) + (1-ex.label) * std::log(1-y + 1e-7f));

                float dOut = error * dsigmoid(y);
                for (int j = 0; j < hiddenSize; ++j) W2[j] -= learningRate * dOut * hidden[j];
                b2 -= learningRate * dOut;

                std::vector<float> dHidden(hiddenSize);
                for (int j = 0; j < hiddenSize; ++j) dHidden[j] = dOut * W2[j] * dsigmoid(hidden[j]);

                for (int i = 0; i < inputSize; ++i)
                    for (int j = 0; j < hiddenSize; ++j)
                        W1[i*hiddenSize + j] -= learningRate * dHidden[j] * x[i];

                for (int j = 0; j < hiddenSize; ++j) b1[j] -= learningRate * dHidden[j];
            }

            std::cout << "Epoch " << epoch << " avg loss: " << (epochLoss / dataset.size()) << "\n";
        }
    }

    int predict(const std::vector<int>& input) const {
        auto x = oneHotEncode(input);
        std::vector<float> hidden;
        float prob = forward(x, hidden);
        std::cout << "Probability: " << prob << "\n";
        return (prob >= 0.5f) ? 1 : 0;
    }

    void save(const std::string& filename) const {
        json j;
        j["W1"] = W1; j["b1"] = b1; j["W2"] = W2; j["b2"] = b2;
        std::ofstream f(filename); f << j.dump(4);
    }

    void load(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) throw std::runtime_error("Cannot open weights file: " + filename);
        json j; f >> j;
        W1 = j["W1"].get<std::vector<float>>();
        b1 = j["b1"].get<std::vector<float>>();
        W2 = j["W2"].get<std::vector<float>>();
        b2 = j["b2"].get<float>();
    }
};

// ------------------ Main ------------------
int main() {
    try {
        fs::create_directory("data");

        std::string datasetFile = "data/rust_detection_dataset.json";
        std::string datasetURL = "https://raw.githubusercontent.com/cairodevv/Jensen/refs/heads/main/datasets/rust_detection_dataset.json";
        std::string weightsFile = "data/weights.json";

        if (!fs::exists(datasetFile)) {
            std::cout << "Downloading dataset...\n";
            downloadDataset(datasetURL, datasetFile);
        }

        auto dataset = loadDataset(datasetFile);
        if (dataset.empty()) { std::cerr << "Dataset empty!\n"; return 1; }

        int inputSize = MAX_INPUT_LEN * CHAR_COUNT;
        int hiddenSize = 64;
        float learningRate = 0.01f;

        NeuralNet nn(inputSize, hiddenSize, learningRate);

        if (fs::exists(weightsFile)) {
            std::cout << "Loading saved weights...\n";
            nn.load(weightsFile);
        } else {
            std::cout << "Training network...\n";
            nn.train(dataset, 50);
            std::cout << "Saving trained weights...\n";
            nn.save(weightsFile);
        }

        std::string input;
        std::cout << "Enter text to classify: ";
        std::getline(std::cin, input);

        std::vector<int> userInput(MAX_INPUT_LEN, 62);
        for (int i = 0; i < MAX_INPUT_LEN && i < input.size(); ++i) {
            auto it = charClassification.find(input[i]);
            userInput[i] = (it != charClassification.end()) ? it->second : 62;
        }

        int pred = nn.predict(userInput);
        std::cout << "Predicted class: " << pred << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
