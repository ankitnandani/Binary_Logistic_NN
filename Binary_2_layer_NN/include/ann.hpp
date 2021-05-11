#ifndef __ANN_HPP
#define __ANN_HPP

#include <vector>
#include <string>
#include "stdint.h"
#include "datahandler.hpp"
#include "data.hpp"
#include "neuron.hpp"

class ann : public neuron{

	std::vector<neuron *>cells;	//0-4 hidden layer 5th cell - output layer
	std::vector<data *> * train_data;
	std::vector<data *> * test_data;
	std::vector<data *> * valid_data;
	double a;

	public:

	ann();	//initialise neurons of hidden layer
	~ann();//


	void set_train_data(std::vector<data *> *);  //
	void set_test_data(std::vector<data *> *);  //
	void set_valid_data(std::vector<data *> *);  //

	double getRandom(double min,double max); //
	double sigmoid(double);

	void train(int epochs, double alpha);	//initialise weights for cells, iterate fprop,bprop updateweights
	void forward();  //
	void backward();
	void updateWeights();

	void test_performance();
	double valid_performance();


};

#endif
