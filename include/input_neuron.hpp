#ifndef __INPUT_NEURON_HPP
#define __INPUT_NEURON_HPP

#include "common.hpp"

class input_neuron : public neuron{
private:
    double _input_value;
public:
    input_neuron(double value)
    : neuron()
    , _input_value(value)
    {}

    double get_output() override{
        return _input_value;
    }

};

#endif
