#include "neuron.hpp"

neuron::neuron(){}
neuron::~neuron(){}

void neuron::add_input_neuron(std::pair<neuron_ptr,double> input){
    _inputs.push_back(input);
}

void neuron::set_activation_function(std::function<double(double)> fn){
    _activation_function = fn;
}

double neuron::get_output(){
    double sum = 0.0;
    for(auto in : _inputs){
        sum += in.first->get_output()*in.second;
    }

    return _activation_function(sum);
}
