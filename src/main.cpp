#include "ffn.hpp"

#include <random>

double sigmoid(double value){
    return 1.0 / (1.0 + exp(-value));
}

double sigmoid_dx(double value){
    return sigmoid(value) * (1.0 - sigmoid(value));
}

double linear(double value){
    return value;
}

double linear_dx(double value){
    return 1.0;
}

int main(int argc, char* argv[])
{
    blaze::setNumThreads(8);
    
    ffn test_ffn(2, 2, 2, 2, 10);
    test_ffn.set_hidden_activation_function(sigmoid);
    test_ffn.set_hidden_activation_function_dx(sigmoid_dx);
    test_ffn.set_output_activation_function(linear);
    test_ffn.set_output_activation_function_dx(linear_dx);

    blaze::DynamicMatrix<double> input(2,1);
    blaze::DynamicMatrix<double> target(2,1);

    column(input, 0) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.0, 2.1);
    column(target, 0) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.5, 2.7);

    test_ffn.train(input, target);

    return 0;
}
