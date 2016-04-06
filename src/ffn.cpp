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

    blaze::Rand< blaze::DynamicMatrix<double> > rndMat;

    _weights.resize(num_hidden_layers + 1);
    _hidden_activations.resize(num_hidden_layers);

    //The first hidden layer must match the size of the input vector
    _weights[0].resize(hidden_layer_dim, input_size);
    rndMat.randomize(_weights[0], 0.0, 1.0);

    _hidden_activations[0].resize(hidden_layer_dim);

    for(size_t i=1; i<num_hidden_layers; i++){
        _weights[i].resize(hidden_layer_dim, hidden_layer_dim);
        rndMat.randomize(_weights[i], 0.0, 1.0);

        _hidden_activations[i].resize(hidden_layer_dim);
    }

    _weights[_num_hidden_layers].resize(output_size, hidden_layer_dim);
    rndMat.randomize(_weights[_num_hidden_layers], 0.0, 1.0);
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

    blaze::DynamicVector<double, blaze::columnVector> X = column(input, 0);
    blaze::DynamicVector<double, blaze::columnVector> y = column(output, 0);

    const size_t L = _num_hidden_layers;

    double error = 9999.0;
    while(error > 0.01 || error < -0.01){
        std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
        z.resize(_num_hidden_layers);

        z[0] = _weights[0]*X;
        for(size_t i=0; i<z.size(); i++){
            _hidden_activations[0][i] = _hidden_activation_function(z[0][i]);
        }

        for(size_t l=1; l<_num_hidden_layers; l++){
            z[l] = _weights[l]*_hidden_activations[l-1];
            for(size_t i=0; i<z.size(); i++){
                _hidden_activations[l][i] = _hidden_activation_function(z[l][i]);
            }
        }

        blaze::DynamicVector<double, blaze::columnVector> o = _weights[L]*_hidden_activations[L - 1];
        for(size_t i=0; i<o.size(); i++){
            o[i] = _output_activation_function(o[i]);
        }

        blaze::DynamicVector<double, blaze::columnVector> dy = o - y;

        double avg_error = 0.0;
        for(size_t i=0; i<dy.size(); i++){
            avg_error += dy[i];
        }
        error = avg_error/dy.size();
        std::cout << "o: " << o << std::endl;
        std::cout << "y: " << y << std::endl;
        std::cout << "error: " << error << std::endl << std::endl;

        //Backprop error
        std::vector< blaze::DynamicVector<double, blaze::columnVector> > e;
        e.resize(_num_hidden_layers+1);

        e[L] = dy;
        for(size_t i=0; i<e.size(); i++){
            e[L][i] *= _output_activation_function_dx(e[L][i]);
        }

        for(size_t l=L-1; l>=0; l--){
            e[l] = trans(_weights[l+1])*e[l+1];
            for(size_t i=0; i<e[l].size(); i++){
                e[l][i] *= _hidden_activation_function_dx(z[l][i]);
            }

            if(l == 0) break;
        }

        //Update weights
        if(X.size() > 1){
            _weights[0] += -LEARNING_RATE*(X*trans(e[0]));
        }
        else{
            _weights[0] += -LEARNING_RATE*trans(X*trans(e[0]));
        }

        for(size_t l=1; l<=_num_hidden_layers; l++){
            if(e[l].size() > 1){
                _weights[l] += -LEARNING_RATE*(_hidden_activations[l-1]*trans(e[l]));
            }
            else{
                _weights[l] += -LEARNING_RATE*trans(_hidden_activations[l-1]*trans(e[l]));
            }
        }

    }
}
