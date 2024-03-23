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

#define DEFAULT_HIDDEN_LAYER_SIZE 0
#define DEFAULT_HIDDEN_LAYERS_COUNT 1


class NeuralNetwork {
private:
    class Random {
    private:
        std::random_device randomDevice;
        double left, right;
    public:
        explicit Random(const double &left, const double &right) {
            this->left = left;
            this->right = right;
        };

        double operator()() {
            std::mt19937 gen(randomDevice());
            std::uniform_real_distribution<> dis(left, right);
            return dis(gen);
        }
    };

    struct Neuron {
        double weightedSum = 0;
        double activation = 0;
        std::function<double(double)> f_activation;
        std::function<double(double)> df_activation;

        explicit Neuron(
                std::function<double(double)> f_activation,
                std::function<double(double)> df_activation
        ) :
                f_activation(std::move(f_activation)),
                df_activation(std::move(df_activation)) {}

        virtual ~Neuron() = default;
    };

    struct Node {
        std::shared_ptr<Neuron> neuron;
        std::vector<double> weights;

        explicit Node(
                size_t weightsCount,
                const std::function<double(double)> &f_activation,
                const std::function<double(double)> &df_activation,
                const std::shared_ptr<Random> &random
        ) :
                neuron(std::make_shared<Neuron>(f_activation, df_activation)) {
            for (size_t i = 0; i < weightsCount; ++i)
                weights.push_back((*random)());
        }
    };

    struct Layer {
    public:
        std::vector<std::shared_ptr<Node>> nodes;

        explicit Layer(
                size_t layerSize,
                size_t prevLayerSize,
                const std::function<double(double)> &f_activation,
                const std::function<double(double)> &df_activation,
                const std::shared_ptr<Random> &random
        ) {
            this->nodes.reserve(layerSize);
            for (size_t i = 0; i < layerSize; ++i)
                this->nodes.push_back(
                        std::make_shared<Node>(
                                prevLayerSize,
                                f_activation,
                                df_activation,
                                random
                        )
                );
        }
    };

    size_t inputLayerSize;
    size_t outputLayerSize;
    size_t hiddenLayerSize;
    size_t hiddenLayersCount;
    std::shared_ptr<Random> random;
    std::vector<std::shared_ptr<Layer>> layers;

    void initLayer(
            std::shared_ptr<Layer> &layer,
            size_t layerSize,
            size_t prevLayerSize,
            const std::function<double(double)> &f_activation,
            const std::function<double(double)> &df_activation
    ) {
        for (size_t i = 0; i < layerSize; ++i)
            layer = std::make_shared<Layer>(
                    layerSize,
                    prevLayerSize,
                    f_activation,
                    df_activation,
                    random
            );
    }

    std::vector<double> forwardPropagation(const std::vector<double> &X) {
        std::vector<double> currentInput = X;
        std::vector<double> newInput;
        for (auto &layer: layers) {
            newInput.clear();
            for (auto &node: layer->nodes) {
                double sum = 0;
                for (size_t i = 0; i < node->weights.size(); ++i)
                    sum += node->weights[i] * currentInput[i];
                node->neuron->weightedSum = sum;
                newInput.push_back(node->neuron->f_activation(sum));
            }
        }
        return newInput;
    }

    void backPropagation(
            const std::vector<double>& inputs,
            const std::vector<double>& targets,
            const std::vector<double>& outputs,
            double learningRate
    ) {
        // Считаем, что targets и outputs являются векторами, соответствующими целевым и выходным данным всего слоя
        std::vector<double> errors; // Ошибки для выходного слоя
        for (size_t i = 0; i < targets.size(); ++i) {
            double error = outputs[i] - targets[i];
            errors.push_back(error);
        }

        for (int l = layers.size() - 1; l >= 0; --l) {
            Layer& layer = *layers[l];
            std::vector<double> gradients; // Градиенты для текущего слоя
            for (size_t j = 0; j < layer.nodes.size(); ++j) {
                double output = layer.nodes[j]->neuron->activation;
                double gradient = errors[j] * layer.nodes[j]->neuron->df_activation(layer.nodes[j]->neuron->weightedSum);
                gradients.push_back(gradient);
            }

            if (l != 0) {
                Layer& prevLayer = *layers[l - 1];
                for (size_t j = 0; j < layer.nodes.size(); ++j) {
                    for (size_t k = 0; k < layer.nodes[j]->weights.size(); ++k) {
                        double input = prevLayer.nodes[k]->neuron->activation;
                        layer.nodes[j]->weights[k] -= learningRate * gradients[j] * input;
                    }
                }
            } else {
                for (size_t j = 0; j < layer.nodes.size(); ++j) {
                    for (size_t k = 0; k < layer.nodes[j]->weights.size(); ++k) {
                        double input = inputs[k];
                        layer.nodes[j]->weights[k] -= learningRate * gradients[j] * input;
                    }
                }
            }

            // Пересчитываем ошибки для следующего слоя
            if (l != 0) {
                errors.clear();
                for (size_t j = 0; j < layers[l - 1]->nodes.size(); ++j) {
                    double error = 0;
                    for (size_t k = 0; k < layer.nodes.size(); ++k) {
                        error += layer.nodes[k]->weights[j] * gradients[k];
                    }
                    errors.push_back(error);
                }
            }
        }
    }






public:
#pragma region Constructor

    explicit NeuralNetwork(
            size_t inputLayerSize,
            size_t outputLayerSize,
            const std::function<double(double)> &f_activation,
            const std::function<double(double)> &df_activation,
            size_t hiddenLayerSize = DEFAULT_HIDDEN_LAYER_SIZE,
            size_t hiddenLayersCount = DEFAULT_HIDDEN_LAYERS_COUNT,
            std::pair<double, double> weightRange = std::make_pair(0, 0.01)
    ) :
            inputLayerSize(inputLayerSize),
            outputLayerSize(outputLayerSize),
            hiddenLayerSize(hiddenLayerSize ? hiddenLayerSize : inputLayerSize / 2),
            hiddenLayersCount(hiddenLayersCount),
            random(std::make_shared<Random>(weightRange.first, weightRange.second)) {

        this->layers = std::vector<std::shared_ptr<Layer>>(hiddenLayersCount + 2);

        initLayer(
                this->layers[0],
                inputLayerSize,
                size_t(NULL),
                f_activation,
                df_activation
        );

        for (size_t i = 1; i <= hiddenLayersCount; ++i)
            initLayer(
                    this->layers[i],
                    hiddenLayerSize,
                    layers[i - 1]->nodes.size(),
                    f_activation,
                    df_activation
            );
        initLayer(
                this->layers[this->layers.size() - 1],
                outputLayerSize,
                this->layers[this->layers.size() - 2]->nodes.size(),
                f_activation,
                df_activation
        );
    }
#pragma endregion

    /*
    void train(
            const std::vector<std::vector<double>> &X_train,
            const std::vector<double> &Y_train,
            size_t iter = 1000,
            double minLearningRate = 0.1,
            double bias = 0.01,
            const std::function<double(double, double)> &loss = [](double prediction, double real) {
                return 0.5 * std::pow(prediction - real, 2);
            },
            const std::function<double(double, double)> &dloss = [](double prediction, double real) {
                return prediction - real;
            }) {
        for (size_t i = 0; i < iter; ++i) {
            if (!(i % 10)) learningRate = std::max(learningRate / 2, minLearningRate);
                double maxError = 0;
            for (size_t j = 0; j < X_train.size(); ++j) {
                // прямой проход
                std::vector<double> predictions = forwardPropagation(X_train[j]);

                // вычисление ошибки
                double error = loss(predictions[0], Y_train[j]);
                maxError = std::max(maxError, error);

                // обратный проход
                backPropagation();
            }
            std::cout << std::fixed;
            std::cout.precision(10);
            std::cout << "Iteration " << i << ". Learning rate = " << learningRate << ". Error = " << maxError << std::endl;
        }
    }
     */

    // Эта функция уже включена в ваш код, она просто нуждается в небольших изменениях
    void train(
            const std::vector<std::vector<double>> &X_train,
            const std::vector<double> &Y_train,
            size_t iterations = 1000,
            double learningRate = 0.1,
            double lambda = 0.01, // Скорость "забывания" для функционала качества
            const std::function<double(double, double)> &loss = [](double prediction, double real) {
                return 0.5 * std::pow(prediction - real, 2);
            },
            const std::function<double(double, double)> &dloss = [](double prediction, double real) {
                return prediction - real;
            }
    ) {
        double Q = 0; // Начальное значение среднего функционала качества

        for (size_t iter = 0; iter < iterations; ++iter) {
            // Случайный выбор наблюдения из X'
            size_t k = std::rand() % X_train.size(); // Используйте лучший источник случайности вместо std::rand()
            std::vector<double> x_k = X_train[k];
            double y_k = Y_train[k];

            // Прямой проход для выбранного наблюдения
            std::vector<double> prediction = forwardPropagation(x_k);

            // Вычисление функции потерь для наблюдения
            double L_k = loss(prediction[0], y_k);

            // Шаг стохастического градиентного алгоритма
            backPropagation(x_k, {y_k},prediction, learningRate);

            // Пересчет функционала качества
            Q = lambda * L_k + (1 - lambda) * Q;

            // Отображение прогресса
            std::cout << std::fixed;
            std::cout.precision(10);
            if (iter % 10000 == 0)
                std::cout << "Iteration " << iter << ": Loss = " << L_k << ", Average Quality = " << Q << std::endl;
        }
    }


    void print() {
        std::cout << "Neural Network Architecture" << std::endl;
        std::cout << "Number of Layers: " << layers.size() << std::endl;
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i << " with size " << layers[i]->nodes.size() << ":" << std::endl;
            for (size_t j = 0; j < layers[i]->nodes.size(); ++j) {
                std::cout << "\tNeuron " << j << " with weights: ";
                for (double weight: layers[i]->nodes[j]->weights) {
                    std::cout << weight << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    double predict(const Content& a, const Content& b) {
        std::vector<double> X;
        for (unsigned i = 0; i < FEATURES * 2; ++i) {
            i < FEATURES ?
            X.push_back(a.features[i])
                         :
            X.push_back(b.features[i - FEATURES]);
        }
        std::vector<double> prediction = forwardPropagation(X);
        return prediction[0];
    }

};

#endif //OCULUZRECSYSTEST_NEURALNETWORK_H
