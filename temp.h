//
// Created by User on 23.03.2024.
//

#ifndef OCULUZRECSYSTEST_TEMP_H
#define OCULUZRECSYSTEST_TEMP_H

#include <cmath>
#include <iostream>
#include "Content.h"
#include "json.hpp"

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

std::pair<std::vector<std::vector<double>>, std::vector<double>>& loadData(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;

    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;

    for (const auto &item: j["data"]) {
        Content point1 = item["point1"].get<Content>();
        Content point2 = item["point2"].get<Content>();
        double similarity = item["similarity"].get<double>();

        X_train.push_back(std::vector<double>());
        for (unsigned i = 0; i < FEATURES * 2; ++i) {
            i < FEATURES ?
            X_train.back().push_back(point1.features[i])
                         :
            X_train.back().push_back(point2.features[i - FEATURES]);
        }
        Y_train.push_back(similarity);
    }

    return *new std::pair<std::vector<std::vector<double>>, std::vector<double>>{X_train, Y_train};
}

std::pair<std::vector<double>, std::pair<Content, Content>> createTestVector() {
    Content test1 = createRandContent();
    Content test2 = createRandContent();
    std::vector<double> test;
    for (unsigned i = 0; i < 2 * FEATURES; ++i) {
        i < FEATURES ?
        test.push_back(test1.features[i])
                     :
        test.push_back(test2.features[i - FEATURES]);
    }
    return {test, {test1, test2}};
}

#endif //OCULUZRECSYSTEST_TEMP_H

