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

#endif //OCULUZRECSYSTEST_TEMP_H

