#include "ffn.hpp"

#include <random>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

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

double sigmoid_tanh(double value){
    return tanh(value);
}

double tanh_dx(double value){
    return 1.0 - pow(tanh(value), 2.0);
}

int main(int argc, char* argv[])
{
    //Get options
    po::options_description desc("supported options");
    desc.add_options()
        ("nt", po::value<int>()->default_value(8), "Number of OMP threads.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    blaze::setNumThreads(vm["nt"].as<int>());

    ffn test_ffn(2, 2, 2, 6, 1);
    test_ffn.set_hidden_activation_function(sigmoid);
    test_ffn.set_hidden_activation_function_dx(sigmoid_dx);
    test_ffn.set_output_activation_function(linear);
    test_ffn.set_output_activation_function_dx(linear_dx);

    blaze::DynamicMatrix<double> input(2,4);
    blaze::DynamicMatrix<double> target(2,4);

    column(input, 0) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.0, 2.1);
    column(target, 0) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.5, 2.7);

    column(input, 1) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.4, 2.6);
    column(target, 1) = blaze::StaticVector<double, 2UL, blaze::columnVector>(2.3, 2.0);

    column(input, 2) = blaze::StaticVector<double, 2UL, blaze::columnVector>(3.4, 3.6);
    column(target, 2) = blaze::StaticVector<double, 2UL, blaze::columnVector>(3.3, 3.0);

    column(input, 3) = blaze::StaticVector<double, 2UL, blaze::columnVector>(1.4, 1.6);
    column(target, 3) = blaze::StaticVector<double, 2UL, blaze::columnVector>(1.7, 1.1);

    std::cout << "Training..." << std::endl;
    test_ffn.train(input, target);

    std::cout << "Predicting..." << std::endl;
    std::cout << test_ffn.predict(blaze::StaticVector<double,2UL,blaze::columnVector>(3.4, 3.6)) << std::endl;

    return 0;
}
