#include "neuron.hpp"
#include "input_neuron.hpp"
#include "output_neuron.hpp"

#include <random>

double sigmoid(double value){
    return 1.0 / (1.0 + exp(-value));
}

double sigmoid_prime(double value){
    return sigmoid(value) * (1.0 - sigmoid(value));
}

int main(int argc, char* argv[])
{
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    input_neuron_ptr x = std::make_shared<input_neuron>(2.0);
    input_neuron_ptr y = std::make_shared<input_neuron>(1.0);
    input_neuron_ptr t = std::make_shared<input_neuron>(0.0);
    input_neuron_ptr v = std::make_shared<input_neuron>(4.0);

    std::vector<neuron_ptr> hidden_layer1;
    for(size_t i=0; i<6; i++){
        neuron_ptr n = std::make_shared<neuron>();
        n->add_input_neuron(std::make_pair(x, dis(gen)));
        n->add_input_neuron(std::make_pair(y, dis(gen)));
        n->add_input_neuron(std::make_pair(t, dis(gen)));
        n->add_input_neuron(std::make_pair(v, dis(gen)));
        n->set_activation_function(*sigmoid);
        n->set_activation_function_derivative(*sigmoid_prime);

        hidden_layer1.push_back(n);
    }

    std::vector<neuron_ptr> hidden_layer2;
    for(size_t i=0; i<6; i++){
        neuron_ptr n = std::make_shared<neuron>();
        for(auto h1 : hidden_layer1){
            n->add_input_neuron(std::make_pair(h1, dis(gen)));
        }

        n->set_activation_function(*sigmoid);
        n->set_activation_function_derivative(*sigmoid_prime);

        hidden_layer2.push_back(n);
    }

    output_neuron_ptr x_out = std::make_shared<output_neuron>();
    x_out->set_activation_function_derivative(*sigmoid_prime);

    output_neuron_ptr y_out = std::make_shared<output_neuron>();
    y_out->set_activation_function_derivative(*sigmoid_prime);

    output_neuron_ptr t_out = std::make_shared<output_neuron>();
    t_out->set_activation_function_derivative(*sigmoid_prime);

    output_neuron_ptr v_out = std::make_shared<output_neuron>();
    v_out->set_activation_function_derivative(*sigmoid_prime);

    for(auto h2 : hidden_layer2){
        x_out->add_input_neuron(std::make_pair(h2, dis(gen)));
        y_out->add_input_neuron(std::make_pair(h2, dis(gen)));
        t_out->add_input_neuron(std::make_pair(h2, dis(gen)));
        v_out->add_input_neuron(std::make_pair(h2, dis(gen)));
    }

    const double target_x = 3.0;

    double z = 1000.0;
    size_t num_itr = 0;
    while(!(z < 0.01 && z > -0.01)){
        double output = x_out->get_output();
        //std::cout << output << std::endl;

        z = target_x - output;
        x_out->set_error(z);
        x->adjust_weights();
        //for(auto n : hidden_layer1) n->adjust_weights();
        //for(auto n : hidden_layer2) n->adjust_weights();

        std::cout << z << std::endl;
        num_itr++;
    }

    std::cout << "finished " << num_itr << " iterations" << std::endl;
    return 0;
}
