#ifndef __OUTPUT_NEURON_HPP
#define __OUTPUT_NEURON_HPP

#include "common.hpp"

class output_neuron : public neuron{
public:
    output_neuron() : neuron(){}
    ~output_neuron(){}

    /*! Accumulate without running activation function,
        since this is an output neuron
    */
    double get_output() override{
        double sum = 0.0;
        for(auto in : _inputs){
            sum += in.first->get_output()*in.second;
        }

        return sum;
    }

    void set_error(double error){
        _error = error;
    }

    double get_error() override{
        return _error;
    }
};

#endif
