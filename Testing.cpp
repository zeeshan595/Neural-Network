#include "Structure/LinearNetwork.h"
#include "Learning/ParticleSwarmOptimisation.h"
#include "Core/IrisDataSet.h"
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <cstring>
#include <sstream>

using namespace Core;
using namespace Structure;
using namespace Learning;

LinearNetwork network(4, 7, 3, LOGISTIC_SIGMOID, SOFTMAX, CONNECT_ALL);

double meanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights)
{
	network.SetWeights(weights);
	std::vector<double> xValues;
	xValues.resize(network.GetInput());// inputs
	std::vector<double> tValues;
	tValues.resize(network.GetOutput());// targets
	double sumSquaredError = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		// assumes data has x-values followed by y-values
		std::copy(data[i].begin(), data[i].begin() + network.GetInput(), xValues.begin());
		std::copy(data[i].begin() + network.GetInput(), data[i].begin() + network.GetInput() + network.GetOutput(), tValues.begin());
		std::vector<double> yValues = network.Compute(xValues);
		for (unsigned int j = 0; j < yValues.size(); ++j)
			sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

	return sumSquaredError;
}

int main()
{
	ParticleSwarmOptimisation pso(meanSquaredError, network.GetWeights());

	std::vector<std::vector<double> > trainData;
	trainData.resize(150);
	for (int i = 0; i < 150; i++)
	{
		trainData[i].resize(7);
		for (int j = 0; j < 7; j++)
		{
			trainData[i][j] = dataset[i][j];
		}
	}
	pso.Train(trainData, 20, 0.01, 0.005, 500);
	std::cout << "Done Trainning, Now Testing With 5.0,3.0,1.6,0.2" << std::endl;
	std::cout << "Expected Output 1, 0, 0" << std::endl;

	std::vector<double> input;
	input.resize(4);
	input[0] = 5.0;
	input[1] = 3.0;
	input[2] = 1.6;
	input[3] = 0.2;

	std::vector<double> output = network.Compute(input);


	std::cout << "\n\nOUTPUT: " << std::endl;
	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << ", ";
	}

	std::cout << std::endl;
	return 0;
}