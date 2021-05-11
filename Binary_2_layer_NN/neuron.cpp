#include "neuron.hpp"

neuron::neuron()
{
	output= new std::vector<double>;
	weight=new std::vector<double>;
	dW = new std::vector<double>;
	dA = new std::vector<double>;
}

neuron::~neuron()
{}

