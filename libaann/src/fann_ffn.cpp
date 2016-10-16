#include "fann_ffn.hpp"

fann_ffn::fann_ffn(size_t input_size, size_t output_size, size_t num_hidden_layers, size_t hidden_layer_dim, fann_activationfunc_enum act)
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
        fann_set_activation_function_layer(_ann, act, i);
    }
    fann_set_activation_function_layer(_ann, FANN_LINEAR, num_layers - 1);

    fann_set_training_algorithm(_ann, FANN_TRAIN_RPROP);
    fann_set_learning_rate(_ann, 0.25);
    fann_set_activation_steepness_hidden(_ann, 0.5);

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

void fann_ffn::train(fs::path file, fs::path output_file, fs::path test_file){
    //fann_init_weights(_ann, fann_read_train_from_file(file.string().c_str()));
    //fann_randomize_weights(_ann, -0.577350269, 0.577350269);
    //fann_randomize_weights(_ann, -0.5, 0.5);
    fann_randomize_weights(_ann, -0.5, 0.5);

    /*float last_e = 9999999.0;
    fann_train_data* data = fann_read_train_from_file(file.string().c_str());
    for(size_t i = 0; i < 2000; i++) {
        float e = fann_train_epoch(_ann, data);
        float de = e - last_e;
        if(e > last_e) {
            std::cout << "Breaking at epoch " << i << " with MSE " << e << std::endl;
            break;
        } else if(fabs(de) < 1e-6) {
            float rate = fann_get_learning_rate(_ann);
            fann_set_learning_rate(_ann, rate * 1.1f);
            std::cout << "updated learning rate [de = " << de << "]" << std::endl;
        }
        last_e = e;

        if(i % 10 == 0) std::cout << " Epoch: " << i << " " << "current error: " << e << std::endl;
    }*/
    fann_train_data* data = fann_read_train_from_file(file.string().c_str());
    fann_train_data* test_data = fann_read_train_from_file(test_file.string().c_str());
    float last_test_error = 99999.9f;
    for(size_t i = 0; i < 2000; i++) {
    //int i = 0;
    //for(;;) {
        float e = fann_train_epoch(_ann, data);
        float test_error = fann_test_data(_ann, test_data);
        /*if(test_error > last_test_error) {
            std::cout << "Breaking at epoch " << i << " with MSE " << e << ", test error " << test_error << std::endl;
            break;
        } else {
            last_test_error = test_error;
        }
        if(test_error < 0.01) {
            std::cout << "Breaking at epoch " << i << " with MSE " << e << ", test error " << test_error << std::endl;
            break;
        }*/

        if(i % 10 == 0) {
             std::cout << " Epoch: " << i << " " << "current training error: " << e << " | testing error: " << test_error << std::endl;
        }

        //i++;
    }

    fann_save(_ann, output_file.string().c_str());
}

double fann_ffn::test(fs::path file) {
    fann_train_data* data = fann_read_train_from_file(file.string().c_str());
    return fann_test_data(_ann, data);
}

float* fann_ffn::predict(float* input){
    return fann_run(_ann, input);
}
