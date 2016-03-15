#ifndef __NEURON_HPP
#define __NEURON_HPP

#include "common.hpp"

class neuron{
protected:
    std::vector<std::pair<neuron_ptr,double>> _inputs; //TODO: map?
    std::function<double(double)> _activation_function;

public:
    neuron();
    virtual ~neuron();

    void add_input_neuron(std::pair<neuron_ptr,double> input);
    void set_activation_function(std::function<double(double)> fn);

    virtual double get_output();

};

#endif
