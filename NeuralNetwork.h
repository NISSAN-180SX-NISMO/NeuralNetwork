//
// Created by User on 22.03.2024.
//

#ifndef OCULUZRECSYSTEST_NEURALNETWORK_H
#define OCULUZRECSYSTEST_NEURALNETWORK_H

#include <functional>
#include <memory>
#include <random>
#include <iostream>
#include <iomanip>
#include <utility>

#include "Eigen/Dense"

#define print(a) std::cout << a << std::endl

using Eigen::MatrixXd;
using namespace std;

#define sout(x) std::cout << "im here " << x << std::endl

class NeuralNetwork {
private:
    vector<MatrixXd> weights;                // Вектор матриц весов для каждого слоя
    vector<MatrixXd> biases;                // Вектор векторов смещения для каждого слоя
    vector<MatrixXd> wd_amounts;           // Вектор векторов взвешенных сумм для каждого слоя
    vector<MatrixXd> activations;           // Вектор векторов активаций для каждого слоя
    function<double(double)> f_activation;    // Функция активации
    function<double(double)> df_activation;    // Функция активации
    function<double(double, double)> f_loss;  // Функция потерь
public:
    vector<MatrixXd> x_train;                // Вектор векторов обучающих данных
    vector<double> y_train;

    void setX(const vector<vector<double>> &X_train) {
        for (auto &x: X_train) {
            x_train.emplace_back(x.size(), 1);
            for (size_t j = 0; j < x.size(); ++j) {
                x_train.back()((long long) j, 0) = x[j];
            }
        }
    }

    void setY(const vector<double> &Y_train) {
        y_train.resize(Y_train.size());
        for (size_t i = 0; i < Y_train.size(); ++i) {
            y_train[i] = Y_train[i];
        }
    }

public:
    MatrixXd forwardPropagation(const MatrixXd &input) {
        MatrixXd result = input;
        activations.clear();
        wd_amounts.clear();

        activations.emplace_back(result);
        for (size_t i = 0; i < weights.size(); ++i) {
            wd_amounts.emplace_back(result * weights[i] + biases[i]);
            activations.emplace_back(wd_amounts.back().unaryExpr(f_activation));
            result = activations.back();
        }
        return result;
    }

    std::pair<MatrixXd, MatrixXd> calcWeightsAndBiasesGradient(const MatrixXd &_output, const MatrixXd &_input, const MatrixXd &_wd_amount) {
        auto bieasesGradient = (_output.array() * _wd_amount.unaryExpr(df_activation).array()).matrix();

        auto weightsGradient = _input.transpose() * bieasesGradient;
        return {weightsGradient, bieasesGradient};
    }

    MatrixXd calcInputGradient(const MatrixXd &biasesGradient, const MatrixXd &weights) {
        return biasesGradient * weights.transpose();
    }

    void backPropagation(const MatrixXd &avgLoss) {
        auto output = avgLoss;
        auto weight = this->weights.end() - 1;
        auto wd_amount = this->wd_amounts.end() - 1;
        auto layer = this->activations.end() - 1;
        for(; layer != activations.begin(); --layer, --weight) {
            auto [weightsGradient, biasesGradient] = calcWeightsAndBiasesGradient(output, *layer,  *(wd_amount));
            auto inputGradient = calcInputGradient(biasesGradient, *weight);
            output = inputGradient;
        }
    }

    explicit NeuralNetwork() {
        f_activation = [](double x) -> double {
            return x * x;
        };
        df_activation = [](double x) -> double {
            return 2 * x;
        };
        MatrixXd w(2, 3);
        w <<
          1, 2, 3,
                1, 2, 3;

        MatrixXd b(1, 3);
        b << 1, 1, 1;

        this->weights.emplace_back(w);
        this->biases.emplace_back(b);
    };

    explicit NeuralNetwork(
            const vector<int> &layersSize,
            function<double(double)> f_activation,
            function<double(double, double)> f_loss
    ) :
            f_activation(std::move(f_activation)),
            f_loss(std::move(f_loss)) {
        for (size_t i = 1; i < layersSize.size(); ++i) {
            auto randomMatrix = MatrixXd::Random(layersSize[i - 1], layersSize[i]);
            auto shiftMatrix = MatrixXd::Constant(layersSize[i - 1], layersSize[i], 1.0);
            auto normalizedWeights = (randomMatrix + shiftMatrix) * 0.01 / 2;
            weights.emplace_back(normalizedWeights);
            biases.emplace_back(MatrixXd::Random(1, layersSize[i]));
        }
    }

    void train(size_t iters, double learningRate, double forgotCoef) {

    }

    void setTrainDataSet(
            const vector<vector<double>> &X_train,
            const vector<double> &Y_train
    ) {
        setX(X_train);
        setY(Y_train);
    }



    double predict(Content content, Content content1) {

    }
};

#endif //OCULUZRECSYSTEST_NEURALNETWORK_H
