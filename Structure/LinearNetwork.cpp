#include "LinearNetwork.h"

using namespace Core;
using namespace Structure;

LinearNetwork::LinearNetwork(int input, int hidden, int output, ActivationType hiddenType, ActivationType outputType, ConnectionType connections)
{
	//Check for errors
	if (input < 1 || hidden < 1 || output < 1)
	{
		std::cout << "input, hidden & output size needs to be 1 or greater" << std::endl;
		throw 20;
	}

	//Setup
	this->input = input;
	this->hidden = hidden;
	this->output = output;
	this->hiddenType = hiddenType;
	this->outputType = outputType;

	//Setup Biases
	hiddenBiases.resize(hidden);
	outputBiases.resize(output);

	//Setup Input->Hidden Weights
	ihweights.resize(input);
	ihweightsDisabled.resize(input);
	for (int i = 0; i < input; i++)
	{
		ihweights[i].resize(hidden);
		ihweightsDisabled[i].resize(hidden);
	}

	hOutputs.resize(hidden);

	//Setup Hidden->Output Weights
	howeights.resize(hidden);
	howeightsDisabled.resize(hidden);
	for (int i = 0; i < hidden; i++)
	{
		howeights[i].resize(output);
		howeightsDisabled[i].resize(output);
	}

	//Generate Random Weights
	GenerateWeights();

	//Setup connections
	ConfigureConnections(connections);
}

std::vector<double> LinearNetwork::Compute(std::vector<double> input)
{
	if (input.size() != (unsigned int)this->input)
	{
		std::cout << "input size does not match" << std::endl;
		throw 20;
	}

	//Setup
	std::vector<double> hiddenValues;
	std::vector<double> outputValues;
	hiddenValues.resize(this->hidden);
	outputValues.resize(this->output);

	for (int i = 0; i < this->hidden; i++)
	{
		hiddenValues[i] = 0;
	}
	for (int i = 0; i < this->output; i++)
	{
		outputValues[i] = 0;
	}
	
	/* Input->Hidden */
	//Weights
	for (int i = 0; i < this->input; i++)
	{
		for (int j = 0; j < this->hidden; j++)
		{
			if (ihweightsDisabled[i][j])
				hiddenValues[j] += ihweights[i][j] * input[i];
		}
	}

	//Biases
	for (int i = 0; i < this->hidden; i++)
	{
		hiddenValues[i] += hiddenBiases[i];
	}

	//Activation
	hOutputs = ActivationMethods::Activation(hiddenValues, hiddenType);

	/* Hidden->Output */
	//Weights
	for (int i = 0; i < this->hidden; i++)
	{
		for (int j = 0; j < this->output; j++)
		{
			if (howeightsDisabled[i][j])
				outputValues[j] += howeights[i][j] * hOutputs[i];
		}
	}

	//Biases
	for (int i = 0; i < this->output; i++)
	{
		outputValues[i] += outputBiases[i];
	}

	//Activation
	outputValues = ActivationMethods::Activation(outputValues, outputType);

	return outputValues;
}

LinearNetwork::~LinearNetwork()
{
	for (int i = 0; i < this->input; i++)
	{
		ihweights[i].erase(ihweights[i].begin(), ihweights[i].begin() + ihweights[i].size());
		ihweightsDisabled[i].erase(ihweightsDisabled[i].begin(), ihweightsDisabled[i].begin() + ihweightsDisabled[i].size());
	}

	for (int i = 0; i < this->hidden; i++)
	{
		howeights[i].erase(howeights[i].begin(), howeights[i].begin() + howeights[i].size());
		howeightsDisabled[i].erase(howeightsDisabled[i].begin(), howeightsDisabled[i].begin() + howeightsDisabled[i].size());
	}

	ihweightsDisabled.erase(ihweightsDisabled.begin(), ihweightsDisabled.begin() + ihweightsDisabled.size());
	howeightsDisabled.erase(howeightsDisabled.begin(), howeightsDisabled.begin() + howeightsDisabled.size());
	hiddenBiases.erase(hiddenBiases.begin(), hiddenBiases.begin() + hiddenBiases.size());
	outputBiases.erase(outputBiases.begin(), outputBiases.begin() + outputBiases.size());
	ihweights.erase(ihweights.begin(), ihweights.begin() + ihweights.size());
	howeights.erase(howeights.begin(), howeights.begin() + howeights.size());
	hOutputs.erase(hOutputs.begin(), hOutputs.begin() + hOutputs.size());
}

std::vector<double> LinearNetwork::GetWeights()
{
	std::vector<double> result;
	result.resize(WeightsLength());
	int k = 0;
	//I->H Weights
	for (int i = 0; i < this->input; i++)
	{
		for (int j = 0; j < this->hidden; j++)
		{
			result[k] = ihweights[i][j];
			k++;
		}
	}

	//Hidden Biases
	for (int i = 0; i < this->hidden; i++)
	{
		result[k] = hiddenBiases[i];
		k++;
	}

	//H->O Weights
	for (int i = 0; i < this->hidden; i++)
	{
		for (int j = 0; j < this->output; j++)
		{
			result[k] = howeights[i][j];
			k++;
		}
	}

	//Output Biases
	for (int i = 0; i < this->output; i++)
	{
		result[k] = outputBiases[i];
		k++;
	}
	return result;
}

void LinearNetwork::SetWeights(std::vector<double> weights)
{
	if (weights.size() != (unsigned int)WeightsLength())
	{
		std::cout << "Weights Length does not match Neural Network" << std::endl;
		throw 20;
	}

	int k = 0;
	//I->H Weights
	for (int i = 0; i < this->input; i++)
	{
		for (int j = 0; j < this->hidden; j++)
		{
			ihweights[i][j] = weights[k];
			k++;
		}
	}

	//Hidden Biases
	for (int i = 0; i < this->hidden; i++)
	{
		hiddenBiases[i] = weights[k];
		k++;
	}

	//H->O Weights
	for (int i = 0; i < this->hidden; i++)
	{
		for (int j = 0; j < this->output; j++)
		{
			howeights[i][j] = weights[k];
			k++;
		}
	}

	//Output Biases
	for (int i = 0; i < this->output; i++)
	{
		outputBiases[i] = weights[k];
		k++;
	}
}

int LinearNetwork::WeightsLength()
{
	return (this->input * this->hidden) + (this->hidden * this->output) + this->hidden + this->output;
}
int LinearNetwork::GetInput()
{
	return input;
}
int LinearNetwork::GetOutput()
{
	return output;
}
std::vector<double> LinearNetwork::gethiddenOutputs()
{
	return hOutputs;
}

void LinearNetwork::GenerateWeights()
{
	std::srand(std::time(0));
	int len = WeightsLength();
	std::vector<double> initialWeights;
	initialWeights.resize(len);
	double low = -0.01;
	double high = 0.01;
	for (int i = 0; i < len; i++)
	{
		double randomNumber = (double)rand() / RAND_MAX;
		initialWeights[i] = (high - low) * randomNumber + low;
	}
	SetWeights(initialWeights);
}

void LinearNetwork::ConfigureConnections(ConnectionType type)
{
	ihweightsDisabled = ConnectionMethod::ConnectNodes(ihweightsDisabled, type);
	howeightsDisabled = ConnectionMethod::ConnectNodes(howeightsDisabled, type);
}