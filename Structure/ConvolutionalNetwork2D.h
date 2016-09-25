#ifndef _CONVOLUTIONAL_NETWORK_2D
#define _CONVOLUTIONAL_NETWORK_2D

#include <iostream>
#include <vector>
#include "../Core/Activation.h"
#include "../Core/CommonFunctions.h"
#include "../Core/ConnectNodes.h"
#include "LinearNetwork.h"

using namespace Core;

namespace Structure
{
	class ConvolutionalNetwork2D
	{
	public:
		std::vector<std::vector<LinearNetwork> > networks;
		ConvolutionalNetwork2D(int input, int hidden, int output, int amountX, int amountY, ActivationType hiddenType, ActivationType outputType, ConnectionType connection);
		~ConvolutionalNetwork2D();

		std::vector<std::vector<std::vector<double> > > Compute(std::vector<std::vector<std::vector<double> > > input);
		int GetAmountX();
		int GetAmountY();

	private:
		int amountX;
		int amountY;
	};
};

#endif