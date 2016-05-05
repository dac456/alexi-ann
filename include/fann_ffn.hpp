#ifndef __FANN_FFN_HPP
#define __FANN_FFN_HPP

#include "common.hpp"
#include <fann.h>

class fann_ffn{
private:
    struct fann* _ann;

public:
    fann_ffn(size_t input_size, size_t output_size, size_t num_hidden_layers, size_t hidden_layer_dim);
    ~fann_ffn();

    void train(fs::path file);
    float* predict(float* input);

};

#endif
