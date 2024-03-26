#include <random>
#include <cmath>
#include "Content.h"
#include "NeuralNetwork.h"
//#include "json.hpp"
//#include "fstream"
//#include "temp.h"
#include "Eigen/Core"




int main() {

    srand(time(nullptr));
    /*
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
    */

#pragma region testCode
    auto F = [](double x) -> double {
        return x * x;
    };

    MatrixXd weightedSums(1, 3);
    weightedSums << 0, 0, 0;
    MatrixXd activations(1, 3);
    activations << 0, 0, 0;


    MatrixXd x(1, 2);
    x << 1, 2;

    MatrixXd y(1, 3);
    y << 0, 100, 0;

    MatrixXd h(1, 3);
    h << 0, 0, 0;
    MatrixXd w(2, 3);
    w << 1, 2, 3,
            1, 2, 3;
    MatrixXd b(1, 3);
    b << 1, 1, 1;

    std::cout << "w:\n" << w << std::endl;
    std::cout << "b:\n" << b << std::endl;
    //std::cout << "x:\n" << x << std::endl;

    weightedSums = x * w + b;
    std::cout << "weightedSums:\n" << weightedSums << std::endl;

    activations = weightedSums.unaryExpr(F);
    std::cout << "activations:\n" << activations << std::endl;

    h = activations;

#pragma endregion // testCode

    auto dF = [](double x) -> double {
        return 2 * x;
    };

    auto dLoss = [](double pred, double target) -> double {
        return 2 * (pred - target);
    };

    print("h:");
    print(h);

    print("y:");
    print(y);

    MatrixXd dLossMatrix(1, h.cols());


    for (int j = 0; j < h.cols(); ++j)
        dLossMatrix(0, j) = dLoss(h(0, j), y(0, j));


    MatrixXd de_dh = dLossMatrix;

    print("de_dh:");
    print(de_dh);

    print("weightedSums.unaryExpr(dF):");
    print(weightedSums.unaryExpr(dF));

    // вход поэлементно умноженный на производную функции активации по всем взвешенным суммам:
    MatrixXd de_dt = (de_dh.array() * weightedSums.unaryExpr(dF).array()).matrix();
    print("de_dt:");
    print(de_dt);

    // градиент весов
    MatrixXd de_dw = x.transpose() * de_dt;

    print("x.transpose():");
    print(x.transpose());

    print("de_dw:");
    print(de_dw);

    // градиент смещений
    MatrixXd de_db = de_dt;

    print("de_db:");
    print(de_db);

    // градиент входа

    MatrixXd de_dx = de_dt * w.transpose();

    print("w.transpose():");
    print(w.transpose());


    print("de_dx:");
    print(de_dx);

    NeuralNetwork nn;
    auto output = nn.forwardPropagation(x);

    MatrixXd dLossMatrix1(1, output.cols());


    for (int j = 0; j < output.cols(); ++j)
        dLossMatrix1(0, j) = dLoss(output(0, j), y(0, j));

    nn.backPropagation(dLossMatrix1);


}