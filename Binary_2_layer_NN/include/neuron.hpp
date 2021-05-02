#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <vector>
#include <stdio.h>


class neuron{

	public:
	std::vector<double> * output;
	std::vector<double> * weight;
	std::vector<double> * dW;
	std::vector<double> * dA;
	double bias;
	double db;

	neuron();	//initialise all arrays and weights
	~neuron();

};

#endif
