#include "logistic.hpp"
#include <random>

logistic::logistic()
{}

logistic::~logistic()
{}

void logistic::set_training_data(std::vector<data *> * vect)
/*
Purpose: To initialise training data in the logistic function by passing pointer to 
train data from datahandler library.
Input : pointer to vector train data of data type data object
Output : none
*/
{
	train_data = vect;
}

void logistic::set_test_data(std::vector<data *> * vect)
/*
Purpose: To initialise test data in the logistic function by passing pointer to 
train data from datahandler library.
Input : pointer to vector test data of data type data object
Output : none
*/
{
	test_data = vect;
}

void logistic::set_valid_data(std::vector<data *> *vect)
/*
Purpose: To initialise validation data in the logistic function by passing pointer to 
train data from datahandler library.
Input : pointer to vector valid data of data type data object
Output : none
*/
{
	valid_data = vect;
}

void logistic::train(int epochs, double a)
/*
Purpose: Propogation of data through the neuron - calling forward propogation, updating y_hat calling back_propogation and calling function to update weights and bias. also initialise weights and bias for 1st epoch
*/
{

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(-1.0,1.0);

	for(int j=0; j<4; j++)	//initalise W,bias
	{
		weights.push_back(generateRandomNumber(-1.0,1.0));
		if(j==3)  bias = generateRandomNumber(-1.0,1.0);
	}

	alpha = a;
	for(int i=0; i<epochs; i++)
	{
		forward();
		test_performance();
		backward();
		update_Wb();	
	}
}

double logistic::generateRandomNumber(double min, double max)
{
	double random = (double) rand() / RAND_MAX;
	return min + random * (max - min);
}

void logistic::forward()
/*
Purpose: Forward Propogation of datapoints through neuron, store prediction in y_hat
*/
{
	double temp;
	int pred;
	int correct=0;

	for( data * dp : * train_data)
	{
		temp=0.0;
		for( int j = 0 ; j<4; j++)
		{
			temp += dp->get_feature_vector()->at(j) * weights.at(j);


		}
		y_hat.push_back(sigmoid(temp + bias));

		if(sigmoid(temp + bias) >= 0.5)
		{
			pred=1;
			prediction.push_back(1);
		}
		else
		{ 
			prediction.push_back(0);
			pred=0;
		}

		if(pred == dp->get_num_label()) correct++;

	}


}

double logistic::sigmoid(double a)
/*
Input: double a
Output: double, sigmoid of input a
Purpose: To calculate the sigmoid of a value
*/
{
	return 1.0 / ( 1.0 + exp( -a));
}

void logistic::backward()
/*
Purpose: Backward Propogation, calculation of dW,dB
*/
{
	int range = weights.size();
	double tempdW;



	for( int i =0 ; i<range; i++)	//computing matrix of dW
	{
		tempdW=0.0;
		for (int j=0; j< train_data->size(); j++)
		{
			tempdW += train_data->at(j) ->get_feature_vector()->at(i) *		//X
				(y_hat.at(j) - train_data->at(j)->get_num_label()) * 	//error ( output - actual)
				(y_hat.at(j)) *	//sigma(A) = y_hat
				(1.0 - y_hat.at(j)); //1 -sigma(A)



		}

		tempdW = tempdW / train_data->size();

		dW.push_back(tempdW);
	}

	db=0;
	for(int i=0; i<train_data->size() ; i++)
	{
		db += (prediction.at(i) - train_data->at(i)->get_num_label()) * y_hat.at(i) * (1 - y_hat.at(i));
	}
	
	db = db / train_data->size();
}

void logistic::update_Wb()
{

	for( int i =0 ; i<weights.size(); i++)
	{
		weights.at(i) = weights.at(i) - alpha * dW. at(i);

	}	

	bias = bias - alpha * db;


}

double logistic::test_performance()
{
	int pred;
	int count=0;
	double temp;

	for( data * dp : *test_data)
	{
		temp=0.0;
		for( int j = 0 ; j<4; j++)
		{
			temp += weights.at(j) * dp->get_feature_vector()->at(j);
		}
		y_hat.push_back(sigmoid(temp + bias));

		if(sigmoid(temp+bias) >= 0.5) pred = 1;
		else pred = 0;

		if ( pred == dp->get_num_label() ) count++;
	}

	double performance = count * 100 / test_data->size();

	printf("In current number of epochs, prediction accuracy is : %.2f\n", performance);

	return performance;
}

double logistic::valid_performance()
{
	int pred;
	int count=0;
	double temp;

	for( data * dp :* valid_data)
	{
		temp=0.0;
		for( int j = 0 ; j<4; j++)
		{
			temp += weights.at(j) * dp->get_feature_vector()->at(j);
		}
		y_hat.push_back(sigmoid(temp + bias));

		if(sigmoid(temp+bias) >= 0.5) pred = 1;
		else pred = 0;

		if ( pred == dp->get_num_label() ) count++;
	}

	double performance = count * 100 / valid_data->size();

	printf("In current number of epochs, valid accuracy is : %.2f\n", performance);

	return performance;
}


int main()
{
	datahandler *dh = new datahandler();

	dh->read_csv("./iris.data" , ",");
	dh->split_data();
	
	logistic * lg = new logistic();

	lg->set_training_data(dh->get_train_data());
	lg->set_test_data(dh->get_test_data());
	lg->set_valid_data(dh->get_valid_data());

	lg->train( 48, 0.01);

	double test_performance = lg->test_performance();

	double valid_performance = lg->valid_performance();
	
	
}
