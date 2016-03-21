#include "neuron.hpp"

neuron::neuron()
    : _error(0.0)
{}
neuron::~neuron(){}

void neuron::add_input_neuron(std::pair<neuron_ptr,double> input){
    _inputs.push_back(input);
    input.first->_add_output_neuron(std::make_pair(shared_from_this(), input.second));
}

void neuron::set_activation_function(std::function<double(double)> fn){
    _activation_function = fn;
}

void neuron::set_activation_function_derivative(std::function<double(double)> fn){
    _activation_function_derivative = fn;
}

double neuron::get_output(){
    double sum = 0.0;
    for(auto in : _inputs){
        sum += in.first->get_output()*in.second;
    }

    return _activation_function(sum);
}


double neuron::get_error(){
    double sum = 0.0;
    for(auto out : _outputs){
        sum += out.first->get_error()*out.second;
    }

    _error = sum;
    return _error;
}

void neuron::adjust_weights(){
    size_t idx = 0;
    for(auto& out : _outputs){
        double w = out.second;
        double e = out.first->get_error();
        double v = out.first->get_output();
        double v_prime = out.first->_activation_function_derivative(v);
        double w_prime = w + (LEARNING_RATE*e*v_prime*this->get_output());

        out.first->_inputs[idx].second = w_prime;
        out.second = w_prime;
        out.first->adjust_weights();

        idx++;
    }
}

size_t neuron::get_num_inputs(){
    return _inputs.size();
}

size_t neuron::get_num_outputs(){
    return _outputs.size();
}

void neuron::_add_output_neuron(std::pair<neuron_ptr,double> output){
    _outputs.push_back(output);
}
