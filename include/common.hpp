#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <utility>

#include <blaze/Math.h>
#include <math.h>

#define LEARNING_RATE 0.2

class neuron;
class input_neuron;
class output_neuron;

typedef std::shared_ptr<neuron> neuron_ptr;
typedef std::shared_ptr<input_neuron> input_neuron_ptr;
typedef std::shared_ptr<output_neuron> output_neuron_ptr;

#endif
