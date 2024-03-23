//
// Created by User on 22.03.2024.
//

#ifndef OCULUZRECSYSTEST_CONTENT_H
#define OCULUZRECSYSTEST_CONTENT_H

#include <string>

constexpr unsigned FEATURES = 5;

struct Content {
    std::string title;
    double features[FEATURES];
};

#endif //OCULUZRECSYSTEST_CONTENT_H
