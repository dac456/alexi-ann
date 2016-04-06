#ifndef __FFN_HPP
#define __FFN_HPP

#include "common.hpp"


class ffn{
private:
    size_t _input_size;
    size_t _output_size;

    size_t _num_hidden_layers;
    size_t _hidden_layer_dim;
    size_t _batch_size;

    //blaze::DynamicMatrix<double> _input_weights;
    //blaze::DynamicMatrix<double> _output_weights;

    std::vector< blaze::DynamicMatrix<double> > _weights;
    std::vector< blaze::DynamicVector<double, blaze::columnVector> > _hidden_activations;

    std::function<double(double)> _hidden_activation_function;
    std::function<double(double)> _hidden_activation_function_dx;

    std::function<double(double)> _output_activation_function;
    std::function<double(double)> _output_activation_function_dx;

public:
    ffn(size_t input_size, size_t output_size, size_t num_hidden_layers, size_t hidden_layer_dim, size_t batch_size);

    void set_hidden_activation_function(std::function<double(double)> fn);
    void set_hidden_activation_function_dx(std::function<double(double)> fn);

    void set_output_activation_function(std::function<double(double)> fn);
    void set_output_activation_function_dx(std::function<double(double)> fn);

    bool train(blaze::DynamicMatrix<double> input, blaze::DynamicMatrix<double> output);
};

#endif
