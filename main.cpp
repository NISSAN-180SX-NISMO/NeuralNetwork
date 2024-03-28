#include <random>
#include <cmath>
#include "Content.h"
#include "NeuralNetwork.h"
#include "json.hpp"
#include "fstream"
#include "temp.h"




int main() {

    srand(time(nullptr));

    std::ifstream file("../data.json");
    nlohmann::json j;
    file >> j;

    std::vector<std::pair<Content, Content>> contentPairs;

    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;

    for (const auto &item: j["data"]) {
        Content point1 = item["point1"].get<Content>();
        Content point2 = item["point2"].get<Content>();
        double similarity = item["similarity"].get<double>();

        contentPairs.emplace_back(point1, point2);

        X_train.push_back(std::vector<double>());
        for (unsigned i = 0; i < FEATURES * 2; ++i) {
            i < FEATURES ?
            X_train.back().push_back(point1.features[i])
                         :
            X_train.back().push_back(point2.features[i - FEATURES]);
        }
        Y_train.push_back(similarity);
    }


    NeuralNetwork model({10, 64, 32, 16, 8, 4, 2, 1});
    model.train(10000, X_train, Y_train, 0.0001, 0.1);

    for (unsigned i = 0; i < 10; ++i) {
        Content test1 = createRandContent();
        Content test2 = createRandContent();
        std::vector<double> test;
        for (unsigned i = 0; i < 2 * FEATURES; ++i) {
            i < FEATURES ?
            test.push_back(test1.features[i])
                         :
            test.push_back(test2.features[i - FEATURES]);
        }

        std::cout << "Predicted similarity:\t" << model.predict(test) << std::endl;
        std::cout << "Real similarity:\t" << evklidSimilarity(test1, test2) << std::endl << std::endl;
    }





}