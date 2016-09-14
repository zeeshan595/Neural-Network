#include "ConvolutionalNetwork.h"

using namespace Core;
using namespace Structure;

ConvolutionalNetwork::ConvolutionalNetwork(int input, int hidden, int output, int amount, ActivationType hiddenType, ActivationType outputType, ConnectionType connection)
{
	if (input < 1 || hidden < 1 || output < 1)
	{
		std::cout << "input, hidden & output cannot be lower then 1" << std::endl;
		throw 20;
	}
	if (amount < 2)
	{
		std::cout << "amount cannot be lower then 2" << std::endl;
		throw 20;
	}

	for (int i = 0; i < amount; i++)
	{
		networks.push_back(LinearNetwork(input * 2, hidden, output, hiddenType, outputType, connection));
	}
}

ConvolutionalNetwork::~ConvolutionalNetwork()
{
	
}

std::vector<std::vector<double> > ConvolutionalNetwork::Compute(std::vector<std::vector<double> > input)
{
	if (input.size() != networks.size() + 1 || input[0].size() != (unsigned int)networks[0].GetInput())
	{
		std::cout << "input's length doesn't match the convolutional network" << std::endl;
		throw 20;
	}

	std::vector<std::vector<double> > result;
	result.resize(networks.size());

	for (unsigned int i = 0; i < networks.size(); i++)
	{
		result[i] = networks[i].Compute(CommonFunctions::CombineLayers(input[i], input[i + 1]));
	}

	return result;
}