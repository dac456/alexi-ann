#include "ffn.hpp"

#include <blaze/util/Random.h>

ffn::ffn(size_t input_size, size_t output_size, size_t num_hidden_layers, size_t hidden_layer_dim, size_t batch_size)
     : _input_size(input_size)
     , _output_size(output_size)
     , _num_hidden_layers(num_hidden_layers)
     , _hidden_layer_dim(hidden_layer_dim)
     , _batch_size(batch_size)
{
    assert(num_hidden_layers >= 1);
    const size_t L = num_hidden_layers + 1;

    blaze::Rand< blaze::DynamicMatrix<double> > rndMat;

    _weights.resize(num_hidden_layers + 2);
    _activations.resize(num_hidden_layers + 2);

    //Index zero holds the input layer
    _weights[0].resize(input_size, 1);
    _weights[0] = 1.0;
    _activations[0].resize(input_size);

    //The first hidden layer must match the size of the input vector
    _weights[1].resize(hidden_layer_dim, input_size);
    rndMat.randomize(_weights[1], 0.0, 1.0);

    for(size_t i=2; i<=num_hidden_layers; i++){
        _weights[i].resize(hidden_layer_dim, hidden_layer_dim);
        rndMat.randomize(_weights[i], 0.0, 1.0);
    }

    for(size_t i=1; i<=num_hidden_layers; i++){
        _activations[i].resize(hidden_layer_dim);
    }

    _weights[L].resize(output_size, hidden_layer_dim);
    rndMat.randomize(_weights[L], 0.0, 1.0);

    _activations[L].resize(output_size);
}

void ffn::set_hidden_activation_function(std::function<double(double)> fn){
    _hidden_activation_function = fn;
}

void ffn::set_hidden_activation_function_dx(std::function<double(double)> fn){
    _hidden_activation_function_dx = fn;
}

void ffn::set_output_activation_function(std::function<double(double)> fn){
    _output_activation_function = fn;
}

void ffn::set_output_activation_function_dx(std::function<double(double)> fn){
    _output_activation_function_dx = fn;
}

bool ffn::train(blaze::DynamicMatrix<double> input, blaze::DynamicMatrix<double> output){
    assert(input.capacity() == output.capacity());
    assert(input.rows() == _input_size && output.rows() == _output_size);

    //blaze::DynamicVector<double, blaze::columnVector> X = column(input, 0);
    //blaze::DynamicVector<double, blaze::columnVector> y = column(output, 0);

    const size_t L = _num_hidden_layers + 1;

    const size_t num_epochs = 10000;
    size_t current_epoch = 0;
    while(current_epoch < num_epochs){
        //Create batches
        //size_t num_batches = ceil(double(input.columns())/double(_batch_size));
        //for(size_t batch_num=0; batch_num<num_batches; batch_num++){
        //    for(size_t b=(batch_num*_batch_size); b<((batch_num+1)*_batch_size); b++){
        //        if(b < input.columns()){
                    blaze::DynamicVector<double, blaze::columnVector> X = column(input, 0); //b
                    blaze::DynamicVector<double, blaze::columnVector> y = column(output, 0); //b

                    std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
                    z.resize(L + 1);

                    _activations[0] = X;
                    z[0] = X;

                    /*z[0] = _weights[0]*X;

                    #pragma omp parallel for
                    for(size_t i=0; i<z.size(); i++){
                        _activations[0][i] = _hidden_activation_function(z[0][i]);
                    }*/

                    for(size_t l = 1; l < L; l++){
                        z[l] = _weights[l]*_activations[l-1];

                        #pragma omp parallel for
                        for(size_t i=0; i<z[l].size(); i++){
                            _activations[l][i] = _hidden_activation_function(z[l][i]);
                        }
                    }

                    //blaze::DynamicVector<double, blaze::columnVector> o = _weights[L]*_activations[L - 1];
                    z[L] = _weights[L]*_activations[L-1];

                    #pragma omp parallel for
                    for(size_t i=0; i<z[L].size(); i++){
                        z[L][i] = _output_activation_function(z[L][i]);
                    }

                    blaze::DynamicVector<double, blaze::columnVector> dy = z[L] - y;

                    bool break_epoch = true;
                    for(size_t i=0; i<dy.size(); i++){
                        if(dy[i] > 0.01 || dy[i] < -0.01){
                            break_epoch = false;
                        }
                    }

                    std::cout << dy << std::endl;
                    if(break_epoch) break;

                    //Backprop error
                    std::vector< blaze::DynamicVector<double, blaze::columnVector> > e;
                    e.resize(L + 1);

                    e[L] = dy;

                    #pragma omp parallel for
                    for(size_t i=0; i<e.size(); i++){
                        e[L][i] *= _output_activation_function_dx(e[L][i]);
                    }

                    for(size_t l=L-1; l>=1; l--){
                        e[l] = trans(_weights[l+1])*e[l+1];

                        #pragma omp parallel for
                        for(size_t i=0; i<e[l].size(); i++){
                            e[l][i] *= _hidden_activation_function_dx(z[l][i]);
                        }

                        if(l == 1) break;
                    }
            //    }
            //}

            //Update weights
            for(size_t l=1; l <= L; l++){
                if(e[l].size() ==1 || _activations[0].size() == 1){
                    _weights[l] += -LEARNING_RATE*trans(_activations[l-1]*trans(e[l]));
                }
                else{
                    _weights[l] += -LEARNING_RATE*(_activations[l-1]*trans(e[l]));
                }
            }
    //    }

        current_epoch++;
    }
}
