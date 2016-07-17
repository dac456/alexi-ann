#ifndef __RNN_HPP
#define __RNN_HPP

#include "common.hpp"

class rnn{
private:
    blaze::DynamicVector<double, blaze::columnVector> _squared_error;
    size_t _observed_examples;

    size_t _input_dim;
    size_t _hidden_layer_dim;
    size_t _output_dim;

    blaze::DynamicMatrix<double> _input_weights;
    blaze::DynamicMatrix<double> _hidden_weights;
    blaze::DynamicMatrix<double> _output_weights;

    //blaze::DynamicMatrix<double> _activations;
    //blaze::DynamicMatrix<double> _output;

    std::function<double(double)> _hidden_activation_function;
    std::function<double(double)> _hidden_activation_function_dx;

    std::function<double(double)> _output_activation_function;
    std::function<double(double)> _output_activation_function_dx;

public:
    rnn(size_t input_dim, size_t hidden_layer_dim, size_t output_dim);
    ~rnn();

    void set_hidden_activation_function(std::function<double(double)> fn);
    void set_hidden_activation_function_dx(std::function<double(double)> fn);

    void set_output_activation_function(std::function<double(double)> fn);
    void set_output_activation_function_dx(std::function<double(double)> fn);

    bool train(blaze::DynamicMatrix<double> input, blaze::DynamicMatrix<double> output);
    blaze::DynamicVector<double, blaze::columnVector> predict(blaze::DynamicMatrix<double> input);
};

#endif
