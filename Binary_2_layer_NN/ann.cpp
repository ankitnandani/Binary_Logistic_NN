#include "ann.hpp"
#include "neuron.hpp"
#include "datahandler.hpp"
#include <cmath>

void ann::set_train_data(std::vector<data *> * vector)
{
	train_data = vector;
}
void ann::set_test_data(std::vector<data *> * vector)
{
	test_data = vector;
}
void ann::set_valid_data(std::vector<data *> * vector)
{
	valid_data=vector;
}

double ann::getRandom(double min,double max)
{
	double random = (double) rand() / RAND_MAX;
	return min + random * (max - min);
}

ann::ann()
{
	for(int i=0;i<6; i++)
	{
		cells.push_back(new neuron());
	}

}

ann::~ann()
{}


void ann::train(int epochs, double alpha)
{
	//initialise alpha
	a=alpha;

	//initialise weights
	int ctr=0;
	int features=4;
	for(neuron *n : cells)
	{
		if(ctr==5)
		{
			for(int i=0;i<cells.size()-1;i++)
			{
				n->weight->push_back(getRandom(-1.0,1.0));
			}n->bias=getRandom(-1.0,1.0);
		}
		else
		{
			for(int i=0; i<features;i++)
			{
				n->weight->push_back(getRandom(-1.0,1.0));
			}n->bias=getRandom(-1.0,1.0);
		}
		ctr++;
	}
	
	printf("Initalise Weights...: Success\n");

	for(int i=0; i<epochs;i++)
	{
		printf("----Start of Epoch----\n");
		forward();
		//printf("1. Fprop of all data points in train set \t : Success\n");
		backward();
		//printf("2. Bprop through the network using train set\t : Success\n");
		updateWeights();
		//printf("3. Update Weights using dW and db \t  :Success\n");
		test_performance();
	}

	
}

double ann::sigmoid(double val)
{
	return 1/( 1 + exp(-val));
}

void ann::forward()
{

	double temp;
	//fprop thru hidden layer
	for(neuron *n : cells)
	{
		n->output->clear();		
		for(data *dp : *train_data)
		{
			temp=0.0;
			for(int i=0; i<4; i++)
			{
				temp+=n->weight->at(i) * dp->get_feature_vector()->at(i);
			}
			temp+= n->bias;
			temp=sigmoid(temp);

			n->output->push_back(temp);
		}
	}
	//printf("Propogation through hidden layer..: Success\n");

	//fprop thru output layer
	cells.at(5)->output->clear();
	for(int i=0; i<train_data->size(); i++)
	{

		temp=0.0;
		for(int j=0; j<cells.size()-1;j++)
		{
			temp+=cells.at(j)->output->at(i) * cells.at(5)->weight->at(j);
		}
		temp += cells.at(5)->bias;
		temp=sigmoid(temp);
		cells.at(5)->output->push_back(temp);
	}
	//printf("Propogation through output layer..: Success\n");

	

}

void ann::backward()
{
	double dy_hat;
	double temp_dW;
	for(int i=cells.size()-1; i>=0 ; i--)
	{
		//for output layer
		if(i == cells.size()-1)
		{
			dy_hat=0.0;
			//calculate dy_hat
			for(int j=0; j<cells.at(i)->output->size(); j++)
			{
				dy_hat += train_data->at(j)->get_num_label() / cells.at(i)->output->at(j) -
					(1 - train_data->at(j)->get_num_label()) /
					(1 -cells.at(i)->output->at(j));
			}
			dy_hat *= -1;
			dy_hat = dy_hat/ cells.at(i)->output->size();
			//printf("dy_hat = %f\n" ,dy_hat);
		
			//calculate each dA
			cells.at(i)->dA->clear();
			for(int j=0; j<cells.at(i)->output->size(); j++)
			{
				cells.at(i)->dA->push_back(
					dy_hat * cells.at(i)->output->at(j) * (1 - cells.at(i)->output->at(j))
					);
			}
			//printf("dA first and last values = %f \t %f\n", cells.at(i)->output->at(0), cells.at(i)->output->at(74));
			

			//calculate each dW
			cells.at(i)->dW->clear();
			for( int j=0 ; j<cells.size()-1; j++)
			{
				temp_dW=0.0;
				for(int k=0;k<cells.at(i)->output->size(); k++)
				{
					temp_dW+=cells.at(j)->output->at(k) * cells.at(i)->dA->at(k);
				}
				cells.at(i)->dW->push_back(temp_dW);
			}
			//printf("Calculation of output layer dW ....:Success\n");
			
			
			//calculate db
			temp_dW=0.0;
			for(double val : *cells.at(i)->dA)
			{
				temp_dW +=val;
			}
			cells.at(i)->db= temp_dW;
			//printf("db = %f\n", cells.at(i)->db);

			//printf("Calculation of output layer db .....:Success\n");

			//printf("BackProp through output layer ......:Success\n");
		}

		//for hidden layer
		else
		{
			//calculate dA
			cells.at(i)->dA->clear();
			for(int j = 0; j< cells.at(i)->output->size(); j++)
			{
				cells.at(i)->dA->push_back(
					cells.at(5)->dA->at(j) * 
					cells.at(5)->weight->at(i) *
					cells.at(i)->output->at(j) *
					( 1 - cells.at(i)->output->at(j) )
					);
			}

			//calculate each dW
			cells.at(i)->dW->clear();
			for(int j=0; j<4;j++)
			{
				temp_dW=0.0;
				for(int k=0; k<cells.at(i)->output->size(); k++)
				{
					temp_dW += train_data->at(k)->get_feature_vector()->at(j) *
							cells.at(i)->dA->at(k);
				}
				cells.at(i)->dW->push_back(temp_dW);
			}

			//printf("Calculation of hidden layer neuron dW ....:Success\n");

			//calculate db
			temp_dW=0.0;
			for(double val : *cells.at(i)->dA)
			{
				temp_dW +=val;
			}
			cells.at(i)->db= temp_dW;
			//printf("dW of ith cell is %f \n" , cells.at(i)->db);
			//printf("Calculation of hidden layer neuron db .....:Success\n");
			
		}
	}
	
}

void ann::updateWeights()
{
\
	for( neuron *n : cells)
	{
		for(int i=0; i<n->weight->size(); i++)
		{
			n->weight->at(i) = n->weight->at(i) - a * n->dW->at(i);
		}
		n->bias = n->bias - a * n->db;
	}

}

void ann::test_performance()
{
	double temp;
	int count = 0;
	int result=0;
	//fprop thru hidden layer
	for(neuron *n : cells)
	{
		n->output->clear();
		for(data *dp : *test_data)
		{
			temp=0.0;
			for(int i=0; i<4; i++)
			{
				temp+=n->weight->at(i) * dp->get_feature_vector()->at(i);
			}
			temp+= n->bias;
			temp=sigmoid(temp);
			n->output->push_back(temp);
		}
	}


	//fprop thru output layer
	cells.at(5)->output->clear();
	for(int i=0; i<test_data->size(); i++)
	{

		temp=0.0;
		for(int j=0; j<cells.size()-1;j++)
		{
			temp+=cells.at(j)->output->at(i) * cells.at(5)->weight->at(j);
		}
		temp += cells.at(5)->bias;
		temp=sigmoid(temp);
		cells.at(5)->output->push_back(temp);

		if(temp > 0.5) result = 1;
		else result =0;

		if (result == test_data->at(i)->get_num_label()) count++;			
	}
	printf("4. number of Correct Preds in Test Set = %d\n", (count) );

}

int main()
{
	datahandler *dh=new datahandler();
	dh->read_csv("./iris.data", ",");

	dh->split_data();

	ann *network = new ann();

	network->set_train_data(dh->get_train_data());
	network->set_valid_data(dh->get_valid_data());
	network->set_test_data(dh->get_test_data());

	network->train(50, 0.001);

}

