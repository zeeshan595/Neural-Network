#include "RecurrentNetwork.h"

using namespace Core;
using namespace Structure;

RecurrentNetwork::RecurrentNetwork(int input, int hidden, int output, int amount, ActivationType hiddenType, ActivationType outputType, ConnectionType connection)
{
	if (input < 1 || hidden < 1 || output < 1)
	{
		std::cout << "Input, Hidden & Output cannot be lower then 1" << std::endl;
		throw 20;
	}
	if (amount < 2)
	{
		std::cout << "Amount cannot be lower then 2" << std::endl;
		throw 20;
	}
	networks.push_back(LinearNetwork(input, hidden, output, hiddenType, outputType, connection));
	for (int i = 1; i < amount; i++)
	{
		networks.push_back(LinearNetwork((input + hidden), hidden, output, hiddenType, outputType, connection));
	}
	outputs.resize(amount);
	for (int i = 0; i < amount; i++)
	{
		outputs[i].resize(output);
	}
}

std::vector<double> RecurrentNetwork::Compute(std::vector<std::vector<double> > input)
{
	if (input.size() != networks.size() || input[0].size() != (unsigned int)networks[0].GetInput())
	{
		std::cout << "Input does not match the Recurrent Network" << std::endl;
		throw 20;
	}

	std::vector<std::vector<double> > hOutputs;
	hOutputs.resize(networks.size());

	outputs[0] = networks[0].Compute(input[0]);
	hOutputs[0] = networks[0].GethiddenOutputs();

	for (unsigned int i = 0; i < networks.size(); i++)
	{
		std::vector<double> layers = CommonFunctions::CombineLayers(input[i], hOutputs[i - 1]);
		outputs[i] = networks[i].Compute(layers);
		hOutputs[i] = networks[i].GethiddenOutputs();
	}

	return outputs[outputs.size() - 1];
}