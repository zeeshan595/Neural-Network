#include "GeneticAlgorithm.h"

using namespace Learning;

GeneticAlgorithm::GeneticAlgorithm(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), std::vector<double> weights)
{
	this->meanSquaredError = meanSquaredError;
	this->weights = weights;
	min = -10.0;
	max = +10.0;
}

GeneticAlgorithm::~GeneticAlgorithm()
{
	weights.erase(weights.begin(), weights.end());
	meanSquaredError = NULL;
}

std::vector<double> GeneticAlgorithm::Train(std::vector<std::vector<double> > trainData, int generations, int population, int mutationAmount)
{
	if (population < 2)
	{
		std::cout << "Population is lower then 2 cannot mate" << std::endl;
		throw 20;
	}
	if (population % 2 != 0)
	{
		std::cout << "Population is not multiple of 2" << std::endl;
		throw 20;
	}
	if ((unsigned int)mutationAmount > this->weights.size())
	{
		std::cout << "Mutation amount cannot be greater then weights" << std::endl;
		throw 20;
	}
	std::srand(std::time(0));
	std::vector<std::vector<double> > weights;
	std::vector<double> errors;
	std::vector<int> index;
	int r = 0;

	errors.resize(population);
	weights.resize(population);
	for (int i = 0; i < population; i++)
	{
		weights[i].resize(this->weights.size());
		for (unsigned int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j] = (max - min) * ((double)rand() / RAND_MAX) + min;
		}
	}

	index.resize(population);
	for (unsigned int i = 0; i < index.size(); i++)
	{
		index[i] = i;
	}

	while (r < generations)
	{
		//Check Each Error
		for (int i = 0; i < population; i++)
		{
			errors[index[i]] = meanSquaredError(trainData, weights[index[i]]);
		}
		//Re-arrange Arrays with best being first
		for (int i = 0; i < population; i++)
		{
			for (int j = i+1; j < population; j++)
			{
				if (errors[index[i]] > errors[index[j]])
				{
					int temp = index[i];
					index[i] = index[j];
					index[j] = temp;
				}
			}
		}

		//Kill All The Weak!
		for (int i = (population / 2); i < population; i++)
		{
			weights[index[i]] = weights[index[i % 2]];
		}

		//Mate Everyone
		for (int i = 2; i < population / 2; i++)
		{
			std::vector<double> w1;
			std::vector<double> w2;

			w1.resize(weights[index[((i * 2) + 0)]].size());
			w2.resize(weights[index[((i * 2) + 1)]].size());

			for (unsigned int j = 0; j < w1.size(); j++)
			{
				if (j % 2 == 0)
				{
					w1[j] = weights[index[(i * 2) + 0]][j];
					w2[j] = weights[index[(i * 2) + 1]][j];
				}
				else
				{
					w1[j] = weights[index[(i * 2) + 1]][j];
					w2[j] = weights[index[(i * 2) + 0]][j];
				}
			}
			weights[index[(i * 2) + 0]] = w1;
			weights[index[(i * 2) + 1]] = w2;
		}

		//Mutate Randomly
		for (int i = 2; i < population; i++)
		{
			for (int j = 0; j < mutationAmount; j++)
			{
				double multiplier = errors[i];
				double m = ((((double)rand() / RAND_MAX) - 0.5f) * 4) / multiplier;

				weights[i][std::rand() % weights[i].size()] = m * (i + 1);
			}
		}

		if (r != 0)
			std::cout << "\x1b[A";
		
		std::cout << "Progress: " << ((double)r / (double)generations) * 100.0 << std::endl;

		r++;
	}

	return weights[0];
}