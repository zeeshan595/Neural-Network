#include "Structure/LinearNetwork.h"
#include "Structure/ConvolutionalNetwork.h"
#include "Structure/ConvolutionalNetwork2D.h"
#include "Structure/RecurrentNetwork.h"
#include "Learning/ParticleSwarmOptimisation.h"
#include "Learning/GeneticAlgorithm.h"
#include "Core/IrisDataSet.h"
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <cstring>
#include <sstream>
#include <stdlib.h>

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
	std::cout << "STARTING LIBRARY TESTING\n\n";

	std::cout << "Recurrent Network ";

	RecurrentNetwork* recNet = new RecurrentNetwork(4, 7, 3, 5, LOGISTIC_SIGMOID, SOFTMAX, CONNECT_ALL);

	delete recNet;

	std::cout << "\033[1;34m[OK]\033[0m\n";
	std::cout << "Convolutional Network ";

	ConvolutionalNetwork* convNet = new ConvolutionalNetwork(4, 7, 3, 5, LOGISTIC_SIGMOID, SOFTMAX, CONNECT_ALL);
	
	delete convNet;

	std::cout << "\033[1;34m[OK]\033[0m\n";
	std::cout << "Convolutional Network 2D ";

	ConvolutionalNetwork2D* convNet2D = new ConvolutionalNetwork2D(4, 7, 3, 5, 6, LOGISTIC_SIGMOID, SOFTMAX, CONNECT_ALL);

	delete convNet2D;

	std::cout << "\033[1;34m[OK]\033[0m\n";


	//Create trainning data
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
	std::vector<double> input;
	input.resize(4);
	std::vector<double> output;

	//Trainning Tests
	std::cout << "Genetic Algorithm Using Linear Network Structure" << std::endl;
	GeneticAlgorithm* ga = new GeneticAlgorithm(meanSquaredError, network.GetWeights());
	std::cout << "Genetic Algorithm Class Created" << std::endl;
	network.SetWeights(ga->Train(trainData, 1000, 12, 5));
	std::cout << "Done Trainning, Now Testing With 5.0,3.0,1.6,0.2" << std::endl;
	std::cout << "Expected Output 1, 0, 0" << std::endl;
	input[0] = 5.0;
	input[1] = 3.0;
	input[2] = 1.6;
	input[3] = 0.2;
	output = network.Compute(input);

	std::cout << "\nOUTPUT: " << std::endl;
	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << ", ";
	}

	delete ga;

	std::cout << "\n\033[1;34m[OK]\033[0m\n";

	//Reset network's weights
	network.GenerateWeights();

	std::cout << "Particle Swarm Optimisation Using Linear Network Structure" << std::endl;

	ParticleSwarmOptimisation* pso = new ParticleSwarmOptimisation(meanSquaredError, network.GetWeights());
	network.SetWeights(pso->Train(trainData, 12, 0.01, 0.005, 1000));
	std::cout << "Done Trainning, Now Testing With 5.0,3.0,1.6,0.2" << std::endl;
	std::cout << "Expected Output 1, 0, 0" << std::endl;
	input[0] = 5.0;
	input[1] = 3.0;
	input[2] = 1.6;
	input[3] = 0.2;
	output = network.Compute(input);

	std::cout << "\nOUTPUT: " << std::endl;
	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << ", ";
	}

	delete pso;

	std::cout << "\n\033[1;34m[OK]\033[0m\n";

	std::cin.get();

	std::cout << std::endl;
	return 0;
}