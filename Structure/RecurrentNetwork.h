#ifndef _RECURRENT_NETWORK
#define _RECURRENT_NETWORK

#include <iostream>
#include <vector>
#include "../Core/Activation.h"
#include "../Core/ConnectNodes.h"
#include "../Core/CommonFunctions.h"
#include "LinearNetwork.h"

using namespace Core;

namespace Structure
{
	class RecurrentNetwork
	{
	public:
		std::vector<LinearNetwork> networks;

		RecurrentNetwork(int input, int hidden, int output, int amount, ActivationType hiddenType, ActivationType outputType, ConnectionType connection);
		~RecurrentNetwork();
		std::vector<double> Compute(std::vector<std::vector<double> > input);
	
	private:
		std::vector<std::vector<double> > outputs;
	};
};

#endif