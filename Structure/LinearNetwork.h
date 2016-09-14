#ifndef _LINEARNETWORK
#define _LINEARNETWORK

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "../Core/Activation.h"
#include "../Core/ConnectNodes.h"

using namespace Core;

namespace Structure
{
	class LinearNetwork
	{
	public:
		std::vector<double> GetWeights();
		void SetWeights(std::vector<double> weights);
		int WeightsLength();
		int GetInput();
		int GetOutput();
		std::vector<double> GethiddenOutputs();
		LinearNetwork(int input, int hidden, int output, ActivationType hiddenType, ActivationType outputType, ConnectionType connections);
		~LinearNetwork();
		std::vector<double> Compute(std::vector<double> input);
		void ConfigureConnections(ConnectionType type);
		void GenerateWeights();

	private:
		std::vector<std::vector<bool> > ihweightsDisabled;
		std::vector<std::vector<bool> > howeightsDisabled;

		int input;
		int hidden;
		int output;
		std::vector<double> hiddenBiases;
		std::vector<double> outputBiases;
		std::vector<std::vector<double> > ihweights;
		std::vector<std::vector<double> > howeights;
		std::vector<double> hOutputs;

		ActivationType hiddenType;
		ActivationType outputType;
	};
};

#endif