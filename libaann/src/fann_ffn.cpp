#include "fann_ffn.hpp"
#include <map>

#include <random>

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
    fann_set_activation_function_layer(_ann, FANN_SIGMOID_SYMMETRIC, num_layers - 1);

    fann_set_training_algorithm(_ann, FANN_TRAIN_RPROP); //RPROP for dtheta
    fann_set_learning_rate(_ann, 0.25);
    //fann_set_activation_steepness_hidden(_ann, 0.5); //dtheta
    fann_set_activation_steepness_hidden(_ann, 0.5); //dx 0.8

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

struct node {
    size_t from;
    size_t fan_in;
    size_t fan_out;
};

void fann_ffn::train(fs::path file, fs::path output_file, fs::path test_file){
    //std::default_random_engine generator;
    //std::normal_distribution<double> distribution(0.0,1.0);

    size_t num_conns = fann_get_total_connections(_ann);
    std::cout << "total connections: " << num_conns << std::endl;

    struct fann_connection* conns = (struct fann_connection*)malloc(sizeof(struct fann_connection*)*num_conns);
    std::map<size_t, node> fan;
    typedef std::map<size_t, node>::iterator fan_it;
    for(size_t i = 0; i < num_conns; i++) {
        struct fann_connection c = conns[i];
        if(fan.count(c.to_neuron)) {
            fan[c.to_neuron].fan_in += 1;
        } else {
            //fan[c.to_neuron].from = c.from_neuron;
            fan[c.to_neuron].fan_in = 1;
            fan[c.to_neuron].fan_out = 0;
        }

        if(fan.count(c.from_neuron)) {
            fan[c.from_neuron].fan_out += 1;
        } else {
            fan[c.from_neuron].fan_in = 0;
            fan[c.from_neuron].fan_out = 1;
        }
    }
    for(size_t i = 0; i < num_conns; i++) {
        struct fann_connection c = conns[i];
        double fan_in = static_cast<double>(fan[c.to_neuron].fan_in);
        double fan_out = static_cast<double>(fan[c.to_neuron].fan_out);
        double r = sqrt(6.0/(fan_in + fan_out));
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-r, r);
        fann_set_weight(_ann, c.from_neuron, c.to_neuron, distribution(generator));
    }
    free(conns);

    //fann_init_weights(_ann, fann_read_train_from_file(file.string().c_str()));
    //fann_randomize_weights(_ann, -0.577350269, 0.577350269);
    //fann_randomize_weights(_ann, -1, 1);
    //fann_randomize_weights(_ann, -0.5, 0.5);
    //fann_randomize_weights(_ann, -1.095445115, 1.095445115);
    static struct fann* out;

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
    float min_test_error = 999999.9f;
    for(size_t i = 0; i < 200; i++) {
    //int i = 0;
    //for(;;) {
        float e = fann_train_epoch(_ann, data);
        float test_error = fann_test_data(_ann, test_data);
        if(test_error < min_test_error) {
            out = fann_copy(_ann);
            min_test_error = test_error;
        }
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

    std::cout << "Saving with error " << min_test_error << std::endl;
    fann_save(out, output_file.string().c_str());
}

double fann_ffn::test(fs::path file) {
    fann_train_data* data = fann_read_train_from_file(file.string().c_str());
    return fann_test_data(_ann, data);
}

float* fann_ffn::predict(float* input){
    return fann_run(_ann, input);
}
