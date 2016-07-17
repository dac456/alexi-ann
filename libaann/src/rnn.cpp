#include "rnn.hpp"

#include <blaze/util/Random.h>

inline int32_t t_idx(size_t L, int32_t t){
    if(t == -1){
        return L;
    }
    else{
        return t;
    }
}

rnn::rnn(size_t input_dim, size_t hidden_layer_dim, size_t output_dim)
    : _observed_examples(0)
    , _input_dim(input_dim)
    , _hidden_layer_dim(hidden_layer_dim)
    , _output_dim(output_dim)
{
    _input_weights.resize(hidden_layer_dim, input_dim);
    _hidden_weights.resize(hidden_layer_dim, hidden_layer_dim);
    _output_weights.resize(output_dim, hidden_layer_dim);

    blaze::Rand< blaze::DynamicMatrix<double> > rndMat;
    rndMat.randomize(_input_weights, 0.0, 0.01);
    rndMat.randomize(_hidden_weights, 0.0, 0.01);
    rndMat.randomize(_output_weights, 0.0, 0.01);

    _squared_error.resize(output_dim);
    _squared_error = 0.0;
}

rnn::~rnn(){

}

void rnn::set_hidden_activation_function(std::function<double(double)> fn){
    _hidden_activation_function = fn;
}

void rnn::set_hidden_activation_function_dx(std::function<double(double)> fn){
    _hidden_activation_function_dx = fn;
}

void rnn::set_output_activation_function(std::function<double(double)> fn){
    _output_activation_function = fn;
}

void rnn::set_output_activation_function_dx(std::function<double(double)> fn){
    _output_activation_function_dx = fn;
}

bool rnn::train(blaze::DynamicMatrix<double> input, blaze::DynamicMatrix<double> output){
    const size_t L = 10;

    for(size_t epoch = 0; epoch < 200; epoch++){
        std::cout << "Starting epoch " << epoch << "..." << std::endl;

        for(size_t series = 0; series < floor(input.columns() / L); series++){
            blaze::DynamicMatrix<double> X(input.rows(), L);
            blaze::DynamicMatrix<double> Y(output.rows(), L);

            for(size_t i = 0; i < 10; i++){
                column(X, i) = column(input, (series*L) + i);
                column(Y, i) = column(output, (series*L) + i);
            }

            //blaze::DynamicMatrix<double> activations(_hidden_layer_dim, L+1);


            blaze::DynamicMatrix<double> output(_output_dim, L);

            //Forward pass
            blaze::DynamicMatrix<double> z(_hidden_layer_dim, L+1);
            blaze::DynamicMatrix<double> s(_hidden_layer_dim, L+1);
            column(z, t_idx(L,-1)) = 0.0;
            column(s, t_idx(L,-1)) = 0.0;
            for(int32_t t = 0; t < L; t++){
                column(z, t) = _input_weights*column(X, t) + _hidden_weights*column(s, t_idx(L,t-1));

                #pragma omp parallel for
                for(size_t i = 0; i < _hidden_layer_dim; i++){
                    s(i, t) = _hidden_activation_function(z(i, t));
                }
                //column(activations, t) = column(s, t);

                column(output, t) = _output_weights*column(s, t);
                #pragma omp parallel for
                for(size_t i = 0; i < _output_dim; i++){
                    output(i,t) = _output_activation_function(output(i,t));
                }
            }

            //BPTT
            blaze::DynamicMatrix<double> output_weights_delta(_output_weights.rows(), _output_weights.columns());
            output_weights_delta = 0.0;
            blaze::DynamicMatrix<double> hidden_weights_delta(_hidden_weights.rows(), _hidden_weights.columns());
            hidden_weights_delta = 0.0;
            blaze::DynamicMatrix<double> input_weights_delta(_input_weights.rows(), _input_weights.columns());
            input_weights_delta = 0.0;

            for(int32_t t = L-1; t >= 0; --t){
                blaze::DynamicVector<double, blaze::columnVector> dy = column(output, t) - column(Y, t);
                #pragma omp parallel for
                for(size_t i = 0; i < dy.size(); i++){
                    _squared_error[i] += pow(dy[i], 2.0);
                    _observed_examples++;
                }
                _squared_error /= _observed_examples;

                blaze::DynamicVector<double, blaze::columnVector> E = dy;
                #pragma omp parallel for
                for(size_t i = 0; i < E.size(); i++){
                    E[i] *= _output_activation_function_dx(z(i, t));
                }
                output_weights_delta += E * trans(column(s, t));

                blaze::DynamicVector<double, blaze::columnVector> e = trans(_output_weights)*dy;
                #pragma omp parallel for
                for(size_t i = 0; i < e.size(); i++){
                    e[i] *= _hidden_activation_function_dx(z(i, t));
                }

                for(int32_t tau = t + 1; tau >= std::max(0, t-5); --tau){
                    hidden_weights_delta += e * trans(column(s, t_idx(L,tau-1)));
                    input_weights_delta += e * trans(column(input, tau));

                    e = trans(_hidden_weights)*e;
                    #pragma omp parallel for
                    for(size_t i = 0; i < e.size(); i++){
                        e[i] *= _hidden_activation_function_dx(z(i, t_idx(L,tau-1))); //tau-1 since e is used in next layer (time step)
                    }
                }
            }

            _input_weights -= 0.001 * input_weights_delta;
            _hidden_weights -= 0.001 * hidden_weights_delta;
            _output_weights -= 0.001 * output_weights_delta;
        }

        std::cout << "Error: " << _squared_error << std::endl;
    }
}

blaze::DynamicVector<double, blaze::columnVector> rnn::predict(blaze::DynamicMatrix<double> input){
    const size_t L = input.columns();
    assert(L >= 1);

    blaze::DynamicMatrix<double> output(_output_dim, L);

    //Forward pass
    blaze::DynamicMatrix<double> z(_hidden_layer_dim, L+1);
    blaze::DynamicMatrix<double> s(_hidden_layer_dim, L+1);
    column(z, t_idx(L,-1)) = 0.0;
    column(s, t_idx(L,-1)) = 0.0;
    for(int32_t t = 0; t < L; t++){
        column(z, t) = _input_weights*column(input,t) + _hidden_weights*column(s, t_idx(L,t-1));

        #pragma omp parallel for
        for(size_t i = 0; i < _hidden_layer_dim; i++){
            s(i, t) = _hidden_activation_function(z(i, t));
        }
        //column(activations, t) = column(s, t);

        column(output, t) = _output_weights*column(s, t);
        #pragma omp parallel for
        for(size_t i = 0; i < _output_dim; i++){
            output(i,t) = _output_activation_function(output(i,t));
        }
    }

    //TODO: test different methods of combining output
    std::cout << output << std::endl;
    return column(output, 0);
}
