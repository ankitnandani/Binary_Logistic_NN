#ifndef __LOGISTIC_HPP
#define __LOGISTIC_HPP

#include <fstream>
#include <cmath>
#include "stdint.h"
#include "data.hpp"
#include "datahandler.hpp"
#include <string>
#include <vector>

class logistic
{
	std::vector<double> y_hat;
	std::vector<double> errors; //dA
	std::vector<double> prediction;
	std::vector<double> weights;
	std::vector<double> dW;

	std::vector<data *> * train_data;
	std::vector<data *> * test_data;
	std::vector<data *> * valid_data;

	double bias;
	double db;
	double alpha;
	int epochs;
	int train_size;

	public:	
	logistic();
	~logistic();

	void set_training_data(std::vector<data *> *);
	void set_test_data(std::vector<data *> *);
	void set_valid_data(std::vector<data *> *);

	void train(int epochs, double);
	double generateRandomNumber(double min, double max);

	void forward();
	double sigmoid(double);

	void backward();
	void update_Wb();
	
	double test_performance();
	double valid_performance();
	
		//to use classifier on user input data points.

	
	
};

#endif
