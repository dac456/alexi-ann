#ifndef __OUTPUT_NEURON_HPP
#define __OUTPUT_NEURON_HPP

#include "common.hpp"

class output_neuron : public neuron{
public:
    output_neuron() : neuron(){}
    ~output_neuron(){}
    
    double get_output() override{
        double sum = 0.0;
        for(auto in : _inputs){
            sum += in.first->get_output()*in.second;
        }

        return sum;
    }

};

#endif
