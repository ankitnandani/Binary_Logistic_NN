#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include "stdint.h"
#include <stdio.h>
#include <vector>
#include <string>

class data{
	std::vector<double> *feature_vector;
	std::string label;
	int num_label;

	public:
	data();
	~data();

	void set_feature_vector(std::vector<double> *);
	void append_to_feature_vector(double);
	void set_label(std::string);

	std::vector<double> * get_feature_vector();
	int get_num_label();
};

#endif
