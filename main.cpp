#include <random>
#include <cmath>
#include "Content.h"
#include "NeuralNetwork.h"
#include "json.hpp"
#include "fstream"

Content createRandContent () {
    Content content;
    content.title = "Random content";
    for (double & feature : content.features) {
        feature = (double) rand() / RAND_MAX;
    }
    return content;
}

double evklidDistance(const Content &point1, const Content &point2) {
    double sum_diff = 0.0;
    for (unsigned i = 0; i < FEATURES; ++i) {
        double diff = point1.features[i] - point2.features[i];
        sum_diff += diff * diff;
    }
    return std::sqrt(sum_diff);
}

double evklidSimilarity(const Content &point1, const Content &point2) {
    double distance = evklidDistance(point1, point2);
    double max_distance = std::sqrt(FEATURES);
    return 1.0 - (distance / max_distance);
}

void from_json(const nlohmann::json &j, Content &c) {
    j.at("title").get_to(c.title);
    j.at("features").get_to(c.features);
}

int main() {

    srand(time(0));
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


    auto f = [](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    };

    auto df = [](double x) {
        return std::exp(x) / (std::exp(2 * x) + 2 * std::exp(x) + 1);
    };

    NeuralNetwork model(10, 1, f, df, 10, 3);
    model.train(X_train, Y_train, 100000, 0.0001, 0.1);
    std::cout << "Training finished" << std::endl;
    for (int i = 0; i < 10; ++i) {
        auto a = createRandContent();
        auto b = createRandContent();
        std::cout << "Predicted value: \t" << model.predict(a, b) << std::endl;
        std::cout << "Real value: \t\t" << evklidSimilarity(a, b) << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
    }
    //model.print();

}