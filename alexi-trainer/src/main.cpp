#include "ffn.hpp"
#include "fann_ffn.hpp"
#include "training_set.hpp"

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
        ("trainingset", po::value<std::string>()->required(), "Path to training set.")
        ("batchsize", po::value<int>()->default_value(1))
        ("numhidden", po::value<int>()->default_value(1))
        ("hiddensize", po::value<int>()->default_value(20))
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    blaze::setNumThreads(vm["nt"].as<int>());

    training_set tset_dx(vm["trainingset"].as<std::string>(), DX);
    tset_dx.save_fann_data("./fann_ffn_dx.data");

    training_set tset_dy(vm["trainingset"].as<std::string>(), DY);
    tset_dy.save_fann_data("./fann_ffn_dy.data");

    training_set tset_dtheta(vm["trainingset"].as<std::string>(), DTHETA);
    tset_dy.save_fann_data("./fann_ffn_dtheta.data");

    //std::cout << tset_dx.get_input_set().columns().size() << " " << tset_dy.get_input_set().columns().size() << std::endl;

    fann_ffn ffn_dx(3, 1, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>());
    ffn_dx.train("./fann_ffn_dx.data", "fann_dx.net");

    fann_ffn ffn_dy(3, 1, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>());
    ffn_dy.train("./fann_ffn_dy.data", "fann_dy.net");

    fann_ffn ffn_dtheta(3, 1, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>());
    ffn_dtheta.train("./fann_ffn_dtheta.data", "fann_dtheta.net");

    /*float test[4] = {-21.3466f, -0.245896f, 2.0f, -1.57f};
    float* out = test_ffn.predict(test);
    std::cout << out[0] << " " << out[1] << std::endl;


    ffn test_ffn2(4, 2, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>(), vm["batchsize"].as<int>());
    test_ffn2.set_hidden_activation_function(sigmoid);
    test_ffn2.set_hidden_activation_function_dx(sigmoid_dx);
    test_ffn2.set_output_activation_function(linear);
    test_ffn2.set_output_activation_function_dx(linear_dx);


    std::cout << "Training..." << std::endl;
    test_ffn2.train(tset.get_input_set(), tset.get_target_set());

    std::cout << "Predicting..." << std::endl;
    std::cout << test_ffn2.predict(blaze::StaticVector<double,4UL,blaze::columnVector>(-21.3466f, -0.245896f, 2.0f, -1.57f)) << std::endl;*/
    //std::cout << test_ffn.predict(blaze::StaticVector<double,5UL,blaze::columnVector>(20, 100, -0.0156063, 2.0, 0.0)) << std::endl;

    return 0;
}
