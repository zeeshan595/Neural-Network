#include "Core/Iris.h"
#include "Core/Activation.h"
#include "Core/ActivationType.h"
#include "Structure/LinearNetwork.h"
#include "Learning/PSO.h"

#include <iostream>
#include <vector>

int main()
{
	std::vector<std::vector<double> > train_data(DATA_SIZE);
	for (int i = 0; i < DATA_SIZE; i++)
	{
		train_data[i].resize(LEN);
		for (int j = 0; j < LEN; j++)
		{
			train_data[i][j] = dataset[i][j];
		}
	}

	std::vector<Core::ActivationType> activations(3);
	activations[0] = Core::ActivationType::NONE;
	activations[1] = Core::ActivationType::LOGISTIC_SIGMOID;
	activations[2] = Core::ActivationType::SOFTMAX;
	Structure::LinearNetwork network({ 4, 7, 3 }, activations);

	Learning::ParticleSwarmOptimisation pso(&network);
	pso.Train(train_data, 12, 0.01, 0.05, 1000);

	std::vector<double> output = network.Compute({4.4,2.9,1.4,0.2});
	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << ",";
	}

	std::cin.ignore();
	return 0;
}