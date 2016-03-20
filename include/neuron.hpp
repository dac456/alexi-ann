#ifndef __NEURON_HPP
#define __NEURON_HPP

#include "common.hpp"

class neuron : public std::enable_shared_from_this<neuron>{
protected:
    std::vector<std::pair<neuron_ptr,double>> _outputs;
    std::vector<std::pair<neuron_ptr,double>> _inputs;
    std::function<double(double)> _activation_function;
    std::function<double(double)> _activation_function_derivative;
    double _error;

public:
    neuron();
    virtual ~neuron();

    void add_input_neuron(std::pair<neuron_ptr,double> input);
    void set_activation_function(std::function<double(double)> fn);
    void set_activation_function_derivative(std::function<double(double)> fn);

    virtual double get_output();
    virtual double get_error();

    void adjust_weights();

    size_t get_num_outputs();

protected:
    void _add_output_neuron(std::pair<neuron_ptr,double> input);

};

#endif
