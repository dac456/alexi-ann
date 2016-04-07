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
    rndMat.randomize(_weights[1], 0.0, 0.2);

    //If there are other hidden layers, there are of uniform size
    for(size_t i=2; i<=num_hidden_layers; i++){
        _weights[i].resize(hidden_layer_dim, hidden_layer_dim);
        rndMat.randomize(_weights[i], 0.0, 0.2);
    }

    for(size_t i=1; i<=num_hidden_layers; i++){
        _activations[i].resize(hidden_layer_dim);
    }

    //The output weights (index L) must match the size of the output vector
    _weights[L].resize(output_size, hidden_layer_dim);
    rndMat.randomize(_weights[L], 0.0, 0.2);

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
        size_t num_batches = ceil(double(input.columns())/double(_batch_size));
        std::cout << "num_batches: " << num_batches << std::endl;
        for(size_t batch_num = 0; batch_num < num_batches; batch_num++){
            std::cout << "batch_num: " << batch_num << std::endl;

            //e_sum holds the accumulated error over all inputs in the batch for each layer
            std::vector< blaze::DynamicMatrix<double> > e_sum;
            e_sum.resize(L + 1);
            for(size_t i = 0; i <= L; i++){
                e_sum[i].resize(_weights[i].rows(), _weights[i].columns());
                e_sum[i] = 0.0;
            }

            size_t num_examples = 0; //Track the number of inputs we have actually seen, in case < batch_size
            for(size_t b = (batch_num*_batch_size); b < ((batch_num+1)*_batch_size); b++){
                std::cout << "b: " << b << std::endl;
                if(b < input.columns()){
                    blaze::DynamicVector<double, blaze::columnVector> X = column(input, b);
                    blaze::DynamicVector<double, blaze::columnVector> y = column(output, b);

                    std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
                    z.resize(L + 1);

                    _activations[0] = X;
                    z[0] = X;

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
                        _activations[L][i] = _output_activation_function(z[L][i]);
                    }

                    blaze::DynamicVector<double, blaze::columnVector> dy = _activations[L] - y;

                    std::cout << dy << std::endl;

                    //Backprop error
                    std::vector< blaze::DynamicVector<double, blaze::columnVector> > e;
                    e.resize(L + 1);

                    e[L] = dy;
                    std::cout << "error mag: " << length(e[L]) << std::endl;

                    #pragma omp parallel for
                    for(size_t i=0; i<e[L].size(); i++){
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

                    //Accumulate error for each layer for this input X
                    for(size_t l = 1; l <= L; l++){
                        if(e[l].size() == 1 || _activations[0].size() == 1){
                            e_sum[l] += trans(e[l]*trans(_activations[l-1]));
                        }
                        else{
                            e_sum[l] += e[l]*trans(_activations[l-1]);
                        }
                    }

                    num_examples++;
                }
                else{
                    break;
                }
            }

            //Update weights
            for(size_t l=1; l <= L; l++){
                //if(e[l].size() ==1 || _activations[0].size() == 1){
                    //_weights[l] += -LEARNING_RATE*trans(_activations[l-1]*trans(e[l]));
                    _weights[l] -= (LEARNING_RATE/num_examples)*e_sum[l];
                //}
                //else{
                    //_weights[l] += -LEARNING_RATE*(_activations[l-1]*trans(e[l]));
                //}
                e_sum[l] = 0.0;
            }

        }

        current_epoch++;
    }
}

blaze::DynamicVector<double, blaze::columnVector> ffn::predict(blaze::DynamicVector<double> input){
    assert(input.size() == _input_size);

    const size_t L = _num_hidden_layers + 1;

    std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
    z.resize(L + 1);

    _activations[0] = input;
    z[0] = input;

    for(size_t l = 1; l < L; l++){
        z[l] = _weights[l]*_activations[l-1];

        #pragma omp parallel for
        for(size_t i=0; i<z[l].size(); i++){
            _activations[l][i] = _hidden_activation_function(z[l][i]);
        }
    }

    z[L] = _weights[L]*_activations[L-1];

    #pragma omp parallel for
    for(size_t i=0; i<z[L].size(); i++){
        _activations[L][i] = _output_activation_function(z[L][i]);
    }

    return _activations[L];
}
