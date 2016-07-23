#include "ffn.hpp"

#include <blaze/util/Random.h>
#include <blaze/util/serialization/Archive.h>

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
    _biases.resize(num_hidden_layers + 2);

    //Index zero holds the input layer
    _weights[0].resize(input_size, 1);
    _weights[0] = 1.0;
    _activations[0].resize(input_size);
    _activations[0] = 0.0;
    _biases[0].resize(input_size);
    _biases[0] = 0.0;

    //The first hidden layer must match the size of the input vector
    _weights[1].resize(hidden_layer_dim, input_size);
    rndMat.randomize(_weights[1], 0.0, 1.0);

    //If there are other hidden layers, there are of uniform size
    for(size_t i=2; i<=num_hidden_layers; i++){
        _weights[i].resize(hidden_layer_dim, hidden_layer_dim);
        rndMat.randomize(_weights[i], 0.0, 1.0);
    }

    for(size_t i=1; i<=num_hidden_layers; i++){
        _activations[i].resize(hidden_layer_dim);
        _activations[i] = 0.0;

        _biases[i].resize(hidden_layer_dim);
        _biases[i] = 0.0;
    }

    //The output weights (index L) must match the size of the output vector
    _weights[L].resize(output_size, hidden_layer_dim);
    rndMat.randomize(_weights[L], 0.0, 1.0);

    _activations[L].resize(output_size);
    _activations[L] = 0.0;

    _biases[L].resize(output_size);
    _biases[L] = 0.0;
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
    assert(input.columns() == output.columns());
    assert(input.rows() == _input_size && output.rows() == _output_size);

    const size_t L = _num_hidden_layers + 1;

    std::vector<int> indices;
    for(size_t b = 0; b < input.columns(); b++) indices.push_back(b);

    const size_t num_epochs = 2000;
    size_t current_epoch = 0;
    bool break_epoch = false;
    double rate = 0.01;
    double last_total_error = 0.0;
    while(current_epoch < num_epochs && !break_epoch){
        size_t num_batches = ceil(double(input.columns())/double(_batch_size));

        //Accumulate the average error at the output layer in e_avg
        blaze::DynamicVector<double, blaze::columnVector> e_avg;
        e_avg.resize(_output_size);
        e_avg = 0.0;

        //for(size_t batch_num = 0; batch_num < num_batches; batch_num++){
            //e_sum holds the accumulated error over all inputs in the batch for each layer
            //e_sum_bias holds the the same accumulated error for biases (not multiplied by activation)
            /*std::vector< blaze::DynamicVector<double, blaze::columnVector> > e_sum;
            e_sum.resize(L + 1);

            std::vector< blaze::DynamicVector<double, blaze::columnVector> > e_sum_bias;
            e_sum_bias.resize(L + 1);

            for(size_t i = 0; i <= L; i++){
                e_sum[i].resize(_weights[i].rows(), _weights[i].columns());
                e_sum[i] = 0.0;

                e_sum_bias[i].resize(_biases[i].rows());
                e_sum_bias[i] = 0.0;
            }*/

            /*blaze::DynamicMatrix<double> X;
            X.resize(_input_size, _batch_size);
            X = 0.0;
            blaze::DynamicMatrix<double> y;
            y.resize(_output_size, _batch_size);
            y = 0.0;

            //Generate batch for input
            size_t num_examples = 0; //Track the number of inputs we have actually seen, in case < batch_size
            size_t xb = 0;
            size_t yb = 0;
            for(size_t b = (batch_num*_batch_size); b < ((batch_num+1)*_batch_size); b++){
                if(b < input.columns()){
                    //blaze::DynamicVector<double, blaze::columnVector> X = column(input, b);
                    //blaze::DynamicVector<double, blaze::columnVector> y = column(output, b);
                    column(X, xb) = column(input, b);
                    column(y, yb) = column(output, b);
                    xb++; yb++;
                    num_examples++;
                }
            }*/
            std::random_shuffle(indices.begin(), indices.end());

            for(size_t b = 0; b < input.columns(); b++){
                blaze::DynamicVector<double, blaze::columnVector> X;
                X.resize(_input_size);
                X = column(input, indices[b]);

                blaze::DynamicVector<double, blaze::columnVector> y;
                y.resize(_output_size);
                y = column(output, indices[b]);

                std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
                //std::vector< blaze::DynamicMatrix<double> > z;
                z.resize(L + 1);

                _activations[0] = X;
                z[0] = X;

                for(size_t l = 1; l < L; l++){
                    z[l] = _weights[l]*_activations[l-1] + _biases[l];

                    //for(size_t j = 0; j < _activations[l].columns(); j++){
                        //#pragma omp parallel for
                        for(size_t i=0; i<z[l].size(); i++){
                            _activations[l][i] = _hidden_activation_function(z[l][i]);
                        }
                    //}
                }

                //blaze::DynamicVector<double, blaze::columnVector> o = _weights[L]*_activations[L - 1];
                z[L] = _weights[L]*_activations[L-1] + _biases[L];

                //for(size_t j = 0; j < _activations[L].columns(); j++){
                    //#pragma omp parallel for
                    for(size_t i=0; i<z[L].size(); i++){
                        _activations[L][i] = _output_activation_function(z[L][i]);
                    }
                //}

                blaze::DynamicVector<double, blaze::columnVector> dy = _activations[L] - y;
                //blaze::DynamicMatrix<double> dy = _activations[L] - y;

                //Backprop error
                std::vector< blaze::DynamicVector<double, blaze::columnVector> > e;
                //std::vector< blaze::DynamicMatrix<double> > e;
                e.resize(L + 1);
                for(size_t i = 0; i < e.size(); i++){
                    e[i] = 0.0;
                }

                e[L] = dy;

                //#pragma omp parallel for
                //for(size_t j = 0; j < num_examples; j++){
                //    e_avg += column(e[L], j);
                //}
                e_avg += e[L];

                //for(size_t j = 0; j < e[L].columns(); j++){
                    //#pragma omp parallel for
                    for(size_t i=0; i<e[L].size(); i++){
                        e[L][i] *= _output_activation_function_dx(z[L][i]);
                    }
                //}

                for(size_t l=L-1; l>=1; l--){
                    e[l] = trans(_weights[l+1])*e[l+1];

                    //for(size_t j = 0; j < e[l].columns(); j++){
                        //#pragma omp parallel for
                        for(size_t i=0; i<e[l].size(); i++){
                            e[l][i] *= _hidden_activation_function_dx(z[l][i]);
                        }
                    //}

                    if(l == 1) break;
                }

                //Accumulate error for each layer for each input X
                /*for(size_t l = 1; l <= L; l++){
                    if(e[l].rows() == 1 || _activations[0].rows() == 1){
                        e_sum[l] += trans(e[l]*trans(_activations[l-1]));
                        e_sum_bias[l] += trans(e[l]);
                    }
                    else{
                        e_sum[l] += e[l]*trans(_activations[l-1]);
                        e_sum_bias[l] += e[l];
                    }
                }

                //Update weights
                for(size_t l=1; l <= L; l++){
                    _weights[l] -= (LEARNING_RATE/num_examples)*e_sum[l];
                    _biases[l] -= (LEARNING_RATE/num_examples)*e_sum_bias[l];
                    e_sum[l] = 0.0;
                    e_sum_bias[l] = 0.0;
                }*/
                //Update weights
                for(size_t l = 1; l <= L; l++){
                    if(e[l].size() == 1 || _activations[0].size() == 1){
                        _weights[l] -= rate * (trans(e[l]*trans(_activations[l-1])));
                        //_biases[l] -= LEARNING_RATE * trans(e[l]);
                    }
                    else{
                        _weights[l] -= rate * (e[l]*trans(_activations[l-1]));
                        //_biases[l] -= LEARNING_RATE * e[l];
                    }
                    _biases[l] -= rate * e[l];
                }

            } //for each input
        //} //for each batch

        blaze::DynamicVector<double> e_tot = e_avg;
        e_avg /= input.columns();
        //std::cout << "e avg: " << length(e_avg) << std::endl;

        if(current_epoch == 0){
            last_total_error = length(e_tot);
        }

        if(!(current_epoch % 20)){
            //std::cout << length(e_tot) << std::endl;
            /*if(length(e_tot) > last_total_error){
                rate *= 0.5;
                std::cout << "new rate: " << rate << std::endl;
            }
            else if(length(e_tot) < last_total_error){
                rate *= 2.0;
                std::cout << "new rate: " << rate << std::endl;
            }*/
            last_total_error = length(e_tot);
        }

        if(length(e_tot) < 0.005){
            break_epoch = true;
            std::cout << "breaking at epoch " << current_epoch << std::endl;
            //std::cout << "total error: " << length(e_tot) << std::endl;
        }

        current_epoch++;
    } //for each epoch

    return true;
}

blaze::DynamicVector<double, blaze::columnVector> ffn::predict(blaze::DynamicVector<double, blaze::columnVector> input){
    assert(input.size() == _input_size);

    const size_t L = _num_hidden_layers + 1;

    //std::vector< blaze::DynamicMatrix<double> > z;
    std::vector< blaze::DynamicVector<double, blaze::columnVector> > z;
    z.resize(L + 1);

    /*blaze::DynamicMatrix<double> input_m;
    input_m.resize(input.size(), 1);
    column(input_m, 0) = input;*/

    _activations[0] = input;
    z[0] = input;

    for(size_t l = 1; l < L; l++){
        z[l] = _weights[l]*_activations[l-1] + _biases[l];

        //for(size_t j = 0; j < _activations[l].columns(); j++){
            #pragma omp parallel for
            for(size_t i=0; i<z[l].size(); i++){
                _activations[l][i] = _hidden_activation_function(z[l][i]);
            }
        //}
    }

    //blaze::DynamicVector<double, blaze::columnVector> o = _weights[L]*_activations[L - 1];
    z[L] = _weights[L]*_activations[L-1] + _biases[L];

    //for(size_t j = 0; j < _activations[L].columns(); j++){
        #pragma omp parallel for
        for(size_t i=0; i<z[L].size(); i++){
            _activations[L][i] = _output_activation_function(z[L][i]);
        }
    //}

    //return column(_activations[L], 0);
    return _activations[L];
}
