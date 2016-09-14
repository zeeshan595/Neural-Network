#ifndef _CONVOLUTIONAL_NETWORK
#define _CONVOLUTIONAL_NETWORK

#include <iostream>
#include <vector>
#include "LinearNetwork.h"
#include "../Core/Activation.h"
#include "../Core/ConnectNodes.h"
#include "../Core/CommonFunctions.h"

using namespace Core;

namespace Structure
{
	class ConvolutionalNetwork
	{
	public:
		std::vector<LinearNetwork> networks;

		ConvolutionalNetwork(int input, int hidden, int output, int amount, ActivationType hiddenType, ActivationType outputType, ConnectionType connection);
		~ConvolutionalNetwork();
		std::vector<std::vector<double> > Compute(std::vector<std::vector<double> > input);
	};
};

#endif