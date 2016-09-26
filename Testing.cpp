#include "Structure/LinearNetwork.h"
#include "Learning/PSO.h"
#include "Core/IrisData.h"

#include <iostream>
#include <vector>

double meanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights);
Structure::LinearNetwork network({ 4, 7, 3 }, { Core::ActivationType::NONE, Core::ActivationType::LOGISTIC_SIGMOID, Core::ActivationType::SOFTMAX });

int main()
{
	std::vector<std::vector<double> > trainData(DATA_SIZE);
	for (int i = 0; i < DATA_SIZE; i++)
	{
		trainData[i].resize(LEN);
		for (int j = 0; j < LEN; j++)
		{
			trainData[i][j] = dataset[i][j];
		}
	}

	Learning::ParticleSwarmOptimisation pso(meanSquaredError, network.GetWeightsLength());
	network.SetWeights(pso.Train(trainData, 20, 0.01, 0.25, 1000));

	std::cout << "Press any key to continue..." << std::endl;
	std::cin.ignore();
	return 0;
}

double meanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights)
{
	network.SetWeights(weights);
	std::vector<double> xValues(network.GetLayers()[0]); //Inputs
	std::vector<double> tValues(network.GetLayers()[network.GetLayers().size() - 1]);// targets
	double sumSquaredError = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		// assumes data has x-values followed by y-values
		std::copy(data[i].begin(), data[i].begin() + network.GetLayers()[0], xValues.begin());
		std::copy(data[i].begin() + network.GetLayers()[0], data[i].begin() + network.GetLayers()[0] + network.GetLayers()[network.GetLayers().size() - 1], tValues.begin());
		std::vector<double> yValues = network.Compute(xValues);
		for (unsigned int j = 0; j < yValues.size(); ++j)
			sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

	return sumSquaredError;
}