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

#ifndef HIDDEN_ACTIVATION_FUNC
#ifndef HIDDEN_ACTIVATION_FUNC_DERIVATIVE
#ifndef LEAKY_RELU_SLOPE
#define LEAKY_RELU_SLOPE 0.01
#define HIDDEN_ACTIVATION_FUNC [](double a) -> double {   \
                return std::max(LEAKY_RELU_SLOPE * a, a); \
                }
#define HIDDEN_ACTIVATION_FUNC_DERIVATIVE [](double a) -> double { \
                return a > 0 ? 1 : LEAKY_RELU_SLOPE;               \
                }
#endif
#endif
#endif

#ifndef OUTPUT_ACTIVATION_FUNC
#ifndef OUTPUT_ACTIVATION_FUNC_DERIVATIVE
#define OUTPUT_ACTIVATION_FUNC [](double a) -> double { \
                return 1 / (1 + std::exp(-a));          \
                }
#define OUTPUT_ACTIVATION_FUNC_DERIVATIVE [](double a) -> double {             \
                return OUTPUT_ACTIVATION_FUNC(a) * (1 - OUTPUT_ACTIVATION_FUNC(a)); \
                }
#endif
#endif

#ifndef ROOT_MEAN_SQUARE_ERROR_FUNC
#ifndef ROOT_MEAN_SQUARE_ERROR_FUNC_DERIVATIVE
#define ROOT_MEAN_SQUARE_ERROR_FUNC [](const Eigen::MatrixXd &pred, const Eigen::MatrixXd &target) -> double { \
                if (pred.rows() != target.rows() || pred.cols() != target.cols()) { \
                    throw std::invalid_argument("Matrices must have the same size"); \
                } \
                return (pred - target).array().pow(2).mean(); \
                }
#define ROOT_MEAN_SQUARE_ERROR_FUNC_DERIVATIVE [](const Eigen::MatrixXd &pred, const Eigen::MatrixXd &target) -> double { \
                if (pred.rows() != target.rows() || pred.cols() != target.cols()) { \
                    throw std::invalid_argument("Matrices must have the same size"); \
                } \
                return (2 * (pred - target).array()).mean(); \
                }
#endif
#endif


class NeuralNetwork {
private:
    struct Layer {
        std::shared_ptr<MatrixXd> input;    // Входной вектор-строка
        MatrixXd weights;                   // Матрица весов
        MatrixXd biases;                    // Вектор-строка смещений
        MatrixXd amounts;                   // Вектор-строка взвешенных сумм
        MatrixXd activations;               // Вектор-строка активаций
    };
    std::vector<Layer> layers;              // Вектор слоев
    MatrixXd input;                         // Входной вектор-строка
    MatrixXd x_train;                       // Вектор векторов-строк обучающих входных данных
    MatrixXd y_train;                       // Вектор-строка обучающих выходных данных

    void setXTrain(const std::vector<std::vector<double>> &x_train) {
        // Создаем матрицу нужного размера
        this->x_train = Eigen::MatrixXd(x_train.size(), x_train[0].size());

        // Заполняем матрицу
        for (size_t i = 0; i < x_train.size(); ++i) {
            for (size_t j = 0; j < x_train[0].size(); ++j) {
                this->x_train(i, j) = x_train[i][j];
            }
        }
    }

    void setYTrain(const std::vector<double> &y_train) {
        // Создаем матрицу нужного размера
        this->y_train = Eigen::MatrixXd(1, y_train.size());

        // Заполняем матрицу
        for (size_t i = 0; i < y_train.size(); ++i) {
            this->y_train(0, i) = y_train[i];
        }
    }

public:
    MatrixXd forwardPropagation() {
        auto layerInput = this->input;
        for (auto &layer: layers) {
            layer.amounts = layerInput * layer.weights + layer.biases;
            layer.activations = layer.amounts.unaryExpr(HIDDEN_ACTIVATION_FUNC);
            layerInput = layer.activations;
        }
        //return layers.back().amounts.unaryExpr(OUTPUT_ACTIVATION_FUNC);
        return layers.back().amounts.unaryExpr(HIDDEN_ACTIVATION_FUNC);
    }

    std::pair<MatrixXd, MatrixXd>
    calcWeightsAndBiasesGradient(const MatrixXd &_output, const MatrixXd &_input, const MatrixXd &_wd_amount) {
        auto bieasesGradient = (_output.array() *
                                _wd_amount.unaryExpr(HIDDEN_ACTIVATION_FUNC_DERIVATIVE).array()).matrix();

        auto weightsGradient = _input.transpose() * bieasesGradient;
        return {weightsGradient, bieasesGradient};
    }

    MatrixXd calcInputGradient(const MatrixXd &biasesGradient, const MatrixXd &weights) {
        return biasesGradient * weights.transpose();
    }



    void backPropagation(MatrixXd output, double learningRate, double lambda) {
        for (int i = layers.size() - 1; i >= 0; --i) {
            auto [weightsGradient, biasesGradient] = calcWeightsAndBiasesGradient(output, *(layers[i].input),
                                                                                  layers[i].amounts);
            output = calcInputGradient(biasesGradient, layers[i].weights);

            layers[i].weights -= learningRate * (weightsGradient + lambda * layers[i].weights);
            layers[i].biases -= learningRate * biasesGradient;
        }
    }

    explicit NeuralNetwork(const vector<int> &layersSize) {
        this->input = MatrixXd(1, layersSize[0]);
        for (size_t i = 1; i < layersSize.size(); ++i) {
            auto randomWeights = MatrixXd::Random(layersSize[i - 1], layersSize[i]);
            auto shiftWeights = MatrixXd::Constant(layersSize[i - 1], layersSize[i], 1.0);
            auto normalizedWeights = (randomWeights + shiftWeights) * 0.00001 / 2;

            auto randomBiases = MatrixXd::Random(1, layersSize[i]);
            auto shiftBiases = MatrixXd::Constant(1, layersSize[i], 1.0);
            auto normalizedBiases = (randomBiases + shiftBiases) * 0.00001 / 2;

            this->layers.emplace_back(Layer{
                    std::make_shared<MatrixXd>(i == 1 ? this->input : this->layers.back().activations),
//                    normalizedWeights,
//                    normalizedBiases,
                    MatrixXd::Random(layersSize[i - 1], layersSize[i]),
                    MatrixXd::Random(1, layersSize[i]),
                    MatrixXd(1, layersSize[i]),
                    MatrixXd(1, layersSize[i])

            });
        }
    }

    class Batcher {
    private:
        std::shared_ptr<MatrixXd> x_train;
        std::shared_ptr<MatrixXd> y_train;

        std::vector<int> indices;
        std::random_device randomDevice;
        std::mt19937 generator;
    public:
        Batcher(const MatrixXd &x_train, const MatrixXd &y_train) {
            this->x_train = std::make_shared<MatrixXd>(x_train);
            this->y_train = std::make_shared<MatrixXd>(y_train);

            this->indices.resize(this->x_train->rows());
            std::iota(indices.begin(), indices.end(), 0);
            generator = std::mt19937(randomDevice());
        }

        std::pair<MatrixXd, MatrixXd> getBatch(size_t size) {
            std::shuffle(indices.begin(), indices.end(), generator);
            indices.resize(size);

            Eigen::MatrixXd x_train_batch(size, this->x_train->cols());
            Eigen::MatrixXd y_train_batch(1, size);
            for (size_t i = 0; i < size; ++i) {
                x_train_batch.row(i) = this->x_train->row(indices[i]);
                y_train_batch(0, i) = this->y_train->operator()(0, indices[i]);
            }
            return {x_train_batch, y_train_batch};
        }

        MatrixXd getPredictions(MatrixXd &input, const MatrixXd &x_train_batch,
                                const std::function<MatrixXd()> &forwardPropagation) {
            size_t size = x_train_batch.rows();
            Eigen::MatrixXd predictions(1, size);

            for (size_t i = 0; i < size; ++i) {
                input = x_train_batch.row(i);
                auto prediction = forwardPropagation();
                for (size_t j = 0; j < prediction.cols(); ++j) {
                    predictions(0, i) = prediction(0, j);
                }
            }
            return predictions;
        }

    };

    void train(
            size_t iters,
            std::vector<std::vector<double>> x_train,
            std::vector<double> y_train,
            double learningRate,
            double lambda
    ) {
        setXTrain(x_train);
        setYTrain(y_train);

        Batcher batcher(this->x_train, this->y_train);
        std::cout << std::fixed;
        std::cout.precision(20);

        for (size_t iter = 0; iter < iters; ++iter) {

            auto [x_train_batch, y_train_batch] = batcher.getBatch(100);

            size_t size = x_train_batch.rows();
            Eigen::MatrixXd predictions(1, size);

            for (size_t i = 0; i < size; ++i) {
                this->input = x_train_batch.row(i);
                auto prediction = forwardPropagation();
                for (size_t j = 0; j < prediction.cols(); ++j) {
                    predictions(0, i) = prediction(0, j);
                }
            }

            auto loss = ROOT_MEAN_SQUARE_ERROR_FUNC(predictions, y_train_batch);
            auto lossBatches = (2 * (predictions - y_train_batch).array()).matrix();
            for (size_t i = 0; i < lossBatches.cols(); ++i) {
                backPropagation(lossBatches.col(i), learningRate, lambda);

            }

            std::cout << "Iter: " << iter << " Loss: " << loss << std::endl;
        }
    }

    double predict (const std::vector<double> &x) {
        if (x.size() != layers[0].weights.rows()) {
            throw std::invalid_argument("Input size must be equal to the input layer size");
        }
        for (size_t i = 0; i < x.size(); ++i) {
            input(0, i) = x[i];
        }
        return forwardPropagation()(0, 0);
    }
};

#endif //OCULUZRECSYSTEST_NEURALNETWORK_H
