#include "fann_ffn.hpp"

fann_ffn::fann_ffn(size_t input_size, size_t output_size, size_t num_hidden_layers, size_t hidden_layer_dim)
    : _ann(nullptr)
{
    size_t num_layers = num_hidden_layers + 2;
    unsigned int* layer_dim = new unsigned int[num_layers];
    for(size_t i = 1; i < num_layers - 1; i++){
        layer_dim[i] = hidden_layer_dim;
    }
    layer_dim[0] = input_size;
    layer_dim[num_layers - 1] = output_size;

    _ann = fann_create_standard_array(num_layers, layer_dim);

    fann_set_activation_function_layer(_ann, FANN_LINEAR, 0);
    for(size_t i = 1; i < num_layers - 1; i++){
        fann_set_activation_function_layer(_ann, FANN_LINEAR, i);
    }
    fann_set_activation_function_layer(_ann, FANN_LINEAR, num_layers - 1);

    fann_set_training_algorithm(_ann, FANN_TRAIN_RPROP);
    fann_set_learning_rate(_ann, 0.01);

    delete[] layer_dim;
}

fann_ffn::fann_ffn(fs::path file){
    _ann = fann_create_from_file(file.string().c_str());
}

fann_ffn::~fann_ffn(){
    if(_ann){
        fann_destroy(_ann);
    }
}

void fann_ffn::train(fs::path file, fs::path output_file){
    fann_train_on_file(_ann, file.string().c_str(), 2000, 0, 0.005);
    fann_save(_ann, output_file.string().c_str());
    //fann_destroy(_ann);
}

float* fann_ffn::predict(float* input){
    return fann_run(_ann, input);
}
