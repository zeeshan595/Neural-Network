#ifndef _GENETIC_ALGORITHM
#define _GENETIC_ALGORITHM

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace Learning
{
	class GeneticAlgorithm
	{
	public:
		double min;
		double max;
		//CrossOverType crossOver;
		double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights);
		GeneticAlgorithm(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), std::vector<double> weights);
		~GeneticAlgorithm();

		std::vector<double> Train(std::vector<std::vector<double> > trainData, int generations, int population, int mutationAmount);

	private:
		std::vector<double> weights;
	};
};

#endif