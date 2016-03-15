#include "neuron.hpp"
#include "input_neuron.hpp"
#include "output_neuron.hpp"

int main(int argc, char* argv[])
{
    input_neuron_ptr x = std::make_shared<input_neuron>(2.0);
    input_neuron_ptr y = std::make_shared<input_neuron>(1.0);
    input_neuron_ptr t = std::make_shared<input_neuron>(0.0);
    input_neuron_ptr v = std::make_shared<input_neuron>(4.0);

    std::vector<neuron_ptr> hidden_layer1;
    for(size_t i=0; i<6; i++){
        neuron_ptr n = std::make_shared<neuron>();
        n->add_input_neuron(std::make_pair(x, 0.5));
        n->add_input_neuron(std::make_pair(y, 0.5));
        n->add_input_neuron(std::make_pair(t, 0.5));
        n->add_input_neuron(std::make_pair(v, 0.5));
        n->set_activation_function([](double value)->double{
            return 1.0 / (1.0 + exp(-value));
        });

        hidden_layer1.push_back(n);
    }

    std::vector<neuron_ptr> hidden_layer2;
    for(size_t i=0; i<6; i++){
        neuron_ptr n = std::make_shared<neuron>();
        for(auto h1 : hidden_layer1){
            n->add_input_neuron(std::make_pair(h1, 0.5));
        }

        n->set_activation_function([](double value)->double{
            return 1.0 / (1.0 + exp(-value));
        });

        hidden_layer2.push_back(n);
    }

    output_neuron_ptr x_out = std::make_shared<output_neuron>();
    output_neuron_ptr y_out = std::make_shared<output_neuron>();
    output_neuron_ptr t_out = std::make_shared<output_neuron>();
    output_neuron_ptr v_out = std::make_shared<output_neuron>();

    for(auto h2 : hidden_layer2){
        x_out->add_input_neuron(std::make_pair(h2, 0.5));
        y_out->add_input_neuron(std::make_pair(h2, 0.5));
        t_out->add_input_neuron(std::make_pair(h2, 0.5));
        v_out->add_input_neuron(std::make_pair(h2, 0.5));
    }

    std::cout << x_out->get_output() << std::endl;

    return 0;
}
