#ifndef __DATAHANDLER_HPP
#define __DATAHANDLER_HPP

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

class datahandler{

	std::vector<data *> * raw_data;
	std::vector<data *> * train_data;
	std::vector<data *> * test_data;
	std::vector<data *> * valid_data;

	const double TRAIN_SET_PERCENT = 0.75;
	const double TEST_SET_PERCENT = 0.20;
	const double VALID_SET_PERCENT = 0.05;

	public:
	datahandler();
	~datahandler();

	void read_csv(std::string, std::string);

	void split_data();

	std::vector<data *> * get_train_data();
	std::vector<data *> * get_test_data();
	std::vector<data *> * get_valid_data();

};

#endif

