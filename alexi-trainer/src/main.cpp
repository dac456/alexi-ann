#include "ffn.hpp"
#include "fann_ffn.hpp"
#include "rnn.hpp"
#include "data_preprocessor.hpp"
#include "training_set.hpp"

#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace po = boost::program_options;
namespace fs = boost::filesystem;

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

double softsign(double value){
    return value / (fabs(value) + 1.0);
}

double softsign_dx(double value){
    return 1.0 / pow(fabs(value) + 1, 2.0);
}

double log_semisig(double value){
    if(value > 0){
        return log(value + 1.0);
    }
    else{
        return -log(fabs(value) + 1.0);
    }
}

double log_semisig_dx(double value){
    if(value > 0){
        return 1.0 / (value + 1.0);
    }
    else{
        return -1.0 / (fabs(value) + 1.0);
    }
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

    /*training_set tset_dx(vm["trainingset"].as<std::string>(), DX);
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
    ffn_dtheta.train("./fann_ffn_dtheta.data", "fann_dtheta.net");*/
    std::vector<fs::path> set_paths;
    std::vector<std::string> set_paths_str;
    boost::split(set_paths_str, vm["trainingset"].as<std::string>(), boost::is_any_of(":"));
    for(auto path : set_paths_str){
        std::cout << path << std::endl;
        set_paths.push_back(fs::path(path));
    }

    data_preprocessor preproc(set_paths);
    //preproc.run_processor(AVERAGE);

    training_set tset_dx(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DX);
    tset_dx.save_blaze_data("./rnn_dx");

    training_set tset_dy(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DY);
    tset_dy.save_blaze_data("./rnn_dy");

    training_set tset_dtheta(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DTHETA);
    tset_dtheta.save_blaze_data("./rnn_dtheta");

    rnn rnn_dx(3, 100, 1);
    rnn_dx.set_hidden_activation_function(log_semisig);
    rnn_dx.set_hidden_activation_function_dx(log_semisig_dx);
    rnn_dx.set_output_activation_function(linear);
    rnn_dx.set_output_activation_function_dx(linear_dx);
    rnn_dx.train("./rnn_dx", "./rnn_dx", 200);

    rnn rnn_dy(3, 100, 1);
    rnn_dy.set_hidden_activation_function(log_semisig);
    rnn_dy.set_hidden_activation_function_dx(log_semisig_dx);
    rnn_dy.set_output_activation_function(linear);
    rnn_dy.set_output_activation_function_dx(linear_dx);
    rnn_dy.train("./rnn_dy", "./rnn_dy", 200);

    /*rnn rnn_dtheta(3, 5, 1);
    rnn_dtheta.set_hidden_activation_function(linear);
    rnn_dtheta.set_hidden_activation_function_dx(linear_dx);
    rnn_dtheta.set_output_activation_function(linear);
    rnn_dtheta.set_output_activation_function_dx(linear_dx);
    rnn_dtheta.train("./rnn_dtheta", "./rnn_dtheta", 100);*/
    /*rnn rnn_dtheta(3, 25, 1);
    rnn_dtheta.set_hidden_activation_function(sigmoid);
    rnn_dtheta.set_hidden_activation_function_dx(sigmoid_dx);
    rnn_dtheta.set_output_activation_function(linear);
    rnn_dtheta.set_output_activation_function_dx(linear_dx);
    rnn_dtheta.train("./rnn_dtheta", "./rnn_dtheta", 100);*/

    /*data_preprocessor preproc(set_paths);
    //preproc.run_processor(AVERAGE);
    //preproc.run_processor(THRESHOLD);

    training_set tset_dx(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DX);
    tset_dx.save_fann_data("./fann_ffn_dx.data");

    training_set tset_dy(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DY);
    tset_dy.save_fann_data("./fann_ffn_dy.data");

    training_set tset_dtheta(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), DTHETA);
    tset_dtheta.save_fann_data("./fann_ffn_dtheta.data");

    training_set tset_terrain(preproc.get_frames(), preproc.get_images(), preproc.get_diff_images(), TERRAIN);
    tset_terrain.save_fann_data("./fann_ffn_terrain.data");

    fann_ffn ffn_dx(5, 1, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>());
    ffn_dx.train("./fann_ffn_dx.data", "fann_dx.net");

    fann_ffn ffn_dy(5, 1, vm["numhidden"].as<int>(), vm["hiddensize"].as<int>());
    ffn_dy.train("./fann_ffn_dy.data", "fann_dy.net");

    fann_ffn ffn_dtheta(5, 1, vm["numhidden"].as<int>(), 4, FANN_SIGMOID);
    ffn_dtheta.train("./fann_ffn_dtheta.data", "fann_dtheta.net");

    fann_ffn ffn_terrain(262, 256, 8, 64, FANN_SIGMOID);
    ffn_terrain.train("./fann_ffn_terrain.data", "fann_terrain.net");*/

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
